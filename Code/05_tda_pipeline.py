
import kmapper as km
import sklearn
import networkx as nx
import pandas as pd
import numpy as np
import logging
import warnings
import polars as pl
import polars as pl
import matplotlib.pyplot as plt
from config import Config

# Try importing RAPIDS cuML
try:
    import cuml
    from cuml.manifold import UMAP
    from cuml.cluster import DBSCAN
    HAS_GPU = True
    print("GPU Acceleration Active: RAPIDS cuML loaded.")
except ImportError:
    HAS_GPU = False
    from sklearn.manifold import TSNE as UMAP # Fallback or keep UMAP if installed from umap-learn
    from sklearn.cluster import DBSCAN
    try:
        import umap
        UMAP = umap.UMAP
    except ImportError:
        pass
    print("WARNING: RAPIDS cuML not found. Falling back to CPU. Performance will be degraded.")

logger = logging.getLogger(__name__)

class TDAPipeline:
    def __init__(self, df_features, lens_cols, cluster_cols, output_path="mapper_output.html"):
        self.df_features = df_features
        self.lens_cols = lens_cols
        self.cluster_cols = cluster_cols
        self.output_path = output_path
        
        # Store data as pandas for easy indexing by mapper results
        self.df_pandas = df_features.to_pandas()
        self.df_lens = self.df_pandas[lens_cols]
        self.X_lens = self.df_lens.values
        self.X_cluster = self.df_pandas[cluster_cols].values
        self.ids = df_features["SPEC_ID_SNO"].to_numpy() # Assuming existing column based on original code
        
        self.graph = None

    def run_mapper(self):
        """
        Executes the KeplerMapper pipeline.
        """
        # Initialize Mapper
        mapper = km.KeplerMapper(verbose=1)
        
        # Projection (Lens)
        logger.info("Projecting data (Lens)...")
        # Use RobustScaler to handle outliers
        scaler = sklearn.preprocessing.RobustScaler()
        X_lens_scaled = scaler.fit_transform(self.X_lens)
        
        if HAS_GPU:
            projector = UMAP(n_components=Config.TDA_N_COMPONENTS, n_neighbors=Config.TDA_N_NEIGHBORS, min_dist=Config.TDA_MIN_DIST, random_state=Config.TDA_RANDOM_STATE)
        else:
            projector = sklearn.manifold.Isomap(n_components=Config.TDA_N_COMPONENTS, n_jobs=-1) # CPU fallback
            
        projected_data = mapper.fit_transform(X_lens_scaled, projection=projector)
        
        # Covering & Clustering
        logger.info("Mapping data...")
        if HAS_GPU:
            clusterer = DBSCAN(eps=Config.TDA_EPS, min_samples=Config.TDA_MIN_SAMPLES)
        else:
            clusterer = sklearn.cluster.DBSCAN(eps=Config.TDA_EPS, min_samples=Config.TDA_MIN_SAMPLES)
            
        self.graph = mapper.map(projected_data, self.X_cluster, 
                          clusterer=clusterer, 
                          cover=km.Cover(n_cubes=Config.TDA_RESOLUTION, perc_overlap=Config.TDA_OVERLAP))
        
        # Visualization
        mapper.visualize(self.graph, path_html=self.output_path, title="T-DMAP: Topological Disease Mapping with GPU")
        
        return self.graph, self.ids

    def analyze_topology(self):
        """
        Performs Graph Analysis:
        1. Cycle Detection (H1 Loops)
        2. Super-Responder Identification (High Risk Lens vs Low Cost Outcome)
        """
        if self.graph is None:
            logger.error("Graph is empty. Run run_mapper() first.")
            return [], []

        logger.info("Analyzing Graph topology...")
        
        # Convert KeplerMapper graph to NetworkX
        nx_graph = km.adapter.to_nx(self.graph)
        
        # 1. Cycle Detection
        try:
            # simple_cycles is for directed graphs. Using cycle_basis for undirected structure analysis usually.
            # But kept closer to original code intent if possible.
            # Original code tried simple_cycles then cycle_basis. wrapper does undirected by default usually?
            # km.adapter.to_nx returns a graph that can be treated as undirected for TDA shape usually.
            cycles = nx.cycle_basis(nx_graph)
            logger.info(f"Detected {len(cycles)} topological loops (Potential Chronic Cycles).")
        except Exception as e:
            logger.warning(f"Cycle detection failed: {e}")
            cycles = []

        # 2. Node Profiling & Super-Responder Logic
        node_stats = []
        
        for node_id, member_indices in self.graph['nodes'].items():
            # member_indices are indices into the original X data
            members_df = self.df_pandas.iloc[member_indices]
            
            # Calculate Aggregates for this Node
            avg_cost = members_df['Total_Cost'].mean() if 'Total_Cost' in members_df else 0
            avg_comorbidity = members_df['Comorbidity_Count'].mean() if 'Comorbidity_Count' in members_df else 0
            
            node_stats.append({
                'node_id': node_id,
                'avg_cost': avg_cost,
                'avg_comorbidity': avg_comorbidity,
                'size': len(member_indices)
            })
            
        stats_df = pd.DataFrame(node_stats)
        
        if stats_df.empty:
            return [], cycles

        # Define thresholds (Dynamic or Fixed)
        # High Severity: > 75th percentile of Comorbidity
        # Low Cost: < 25th percentile of Cost
        
        high_sev_thresh = stats_df['avg_comorbidity'].quantile(Config.TDA_HIGH_SEVERITY_Q)
        low_cost_thresh = stats_df['avg_cost'].quantile(Config.TDA_LOW_COST_Q)
        
        super_responder_nodes = stats_df[
            (stats_df['avg_comorbidity'] >= high_sev_thresh) & 
            (stats_df['avg_cost'] <= low_cost_thresh)
        ]
        
        logger.info(f"Identified {len(super_responder_nodes)} Super-Responder Nodes.")
        
        # Collect Patient IDs
        final_ids = []
        for node_id in super_responder_nodes['node_id']:
            member_indices = self.graph['nodes'][node_id]
            patient_ids = self.df_pandas.iloc[member_indices]['SPEC_ID_SNO'].tolist()
            final_ids.extend(patient_ids)
            
        return list(set(final_ids)), cycles

    def save_graph_as_png(self, output_path="tda_graph.png"):
        print(f"   -> Saving static graph image to {output_path}...")
        
        # 1. NetworkX 그래프 변환
        if self.graph is None:
            print("❌ Error: Graph not found. Run run_mapper() first.")
            return
            
        nx_graph = km.adapter.to_nx(self.graph)
        
        # 2. 노드 색상 결정 (중증도 기준)
        # 각 노드(클러스터)에 속한 환자들의 평균 중증도(Comorbidity_Count)를 계산해 색깔로 씀
        node_colors = []
        for node_id in nx_graph.nodes():
            member_ids = self.graph['nodes'][node_id]
            # 해당 노드 환자들의 중증도 평균
            # NOTE: User snippet used 'comorbidity_cnt', passing correct column name 'Comorbidity_Count'
            col_name = 'Comorbidity_Count' if 'Comorbidity_Count' in self.df_lens.columns else 'comorbidity_cnt'
            if col_name not in self.df_lens.columns:
                 # Fallback if neither exists (should not happen based on fe)
                 avg_severity = 0
            else:
                 avg_severity = self.df_lens.iloc[member_ids][col_name].mean()
            node_colors.append(avg_severity)

        # 3. 레이아웃 설정 (Kamada-Kawai가 TDA 모양을 가장 잘 잡아줍니다)
        # 노드가 너무 많으면 spring_layout을 쓰세요.
        try:
            pos = nx.kamada_kawai_layout(nx_graph)
        except:
            pos = nx.spring_layout(nx_graph, k=0.15, iterations=20)

        # 4. 그림 그리기
        plt.figure(figsize=(12, 10)) # 이미지 크기 (인치)
        
        nx.draw_networkx_nodes(
            nx_graph, pos, 
            node_color=node_colors, 
            cmap=plt.cm.Spectral, # 빨강(고위험)~파랑(저위험) 색상테마
            node_size=50,
            alpha=0.8
        )
        
        # 엣지(연결선) 그리기
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.3, edge_color='gray')
        
        plt.title("T-DMAP Topological Disease Map (Color: Severity)", fontsize=15)
        plt.axis('off') # 축 숨기기
        
        # 5. 저장
        plt.savefig(output_path, dpi=300, bbox_inches='tight') # 고해상도 저장
        plt.close()
        print("   -> PNG Saved Successfully!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n[Testing TDA Pipeline Module]")
    
    # Create Dummy Features
    data = {
        "SPEC_ID_SNO": range(100),
        "Total_Cost": np.random.rand(100) * 1000,
        "Visit_Count": np.random.randint(1, 10, 100),
        "Comorbidity_Count": np.random.randint(1, 5, 100),
        "Total_Med_Qty": np.random.rand(100) * 50
    }
    # Add fake drugs
    for i in range(5):
        data[f"Drug_D{i}"] = np.random.randint(0, 2, 100)
        
    df_features = pl.DataFrame(data)
    
    lens_cols = ["Total_Cost", "Visit_Count", "Comorbidity_Count"]
    cluster_cols = [c for c in df_features.columns if c.startswith("Drug_")]
    
    print(f"Features Shape: {df_features.shape}")
    
    try:
        # Instantiate Pipeline
        pipeline = TDAPipeline(df_features, lens_cols, cluster_cols, output_path="test_mapper.html")
        
        graph, ids = pipeline.run_mapper()
        print("Mapper Graph Generated.")
        
        super_responders, cycles = pipeline.analyze_topology()
        print(f"Analysis Complete. Cycles: {len(cycles)}, Super-Responders: {len(super_responders)}")
        
        pipeline.save_graph_as_png("test_graph.png")
        
    except Exception as e:
        print(f"Pipeline Execution Failed (Expected if too little data for UMAP/DBSCAN): {e}")

    print("Test Complete.")
