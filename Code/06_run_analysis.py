
import sys
import os
import importlib
import pandas as pd
from config import Config

# Dynamic imports because python modules starting with numbers cannot be imported directly
dl = importlib.import_module("02_data_loader")
df_mod = importlib.import_module("03_data_filter")
fe = importlib.import_module("04_feature_engine")
tp = importlib.import_module("05_tda_pipeline")

def main():
    print("="*60)
    print(" T-DMAP: Topological Disease Mapping & Analysis Platform")
    print("="*60)

    # 1. Load Data
    # Using functional data loader. Returning 3 DataFrames.
    spec, sick, treat = dl.load_and_merge_data(base_path=Config.DATA_DIR)

    # 2. Filter Data
    # DataFilter class expects (df_merged, df_sick)
    # Refactoring Logic: Merge Spec and Treat to create a 'df_merged' that DataFilter works with.
    # DataFilter uses 'AMT' (from Treat) and 'age_group' (from Spec).
    # NOTE: One Spec has Multiple Treat. Join on SPEC_ID_SNO.
    
    print("[1/4] Preparing Data for Filtering...")
    df_merged = spec.join(treat, on="SPEC_ID_SNO", how="left")
    
    # Apply noise reduction and feature selection
    filter_module = df_mod.DataFilter(df_merged, sick)
    df_merged_filtered, df_sick_filtered = filter_module.apply_filters()
    
    # Post-Filter: We need to pass the filtered data to Feature Engine.
    # Feature Engine expects (spec, sick, treat).
    # We must de-couple df_merged back into spec and treat OR modify how we pass them.
    # Since df_merged_filtered is already filtered by rows, we can use it as both spec and treat sources,
    # OR we can assume FeatureEngine can handle 'df_merged' if we passed it correctly.
    # But FeatureEngine is functional and expected specific inputs.
    
    # Let's map the filtered merge back to treat and spec roughly.
    # Actually, df_merged_filtered HAS the columns for both.
    df_treat_filtered = df_merged_filtered # It has AMT, GNL_NM_CD, etc.
    df_spec_filtered = df_merged_filtered # It has SPEC_ID_SNO, age_group
    
    # 3. Create Features
    # Now we feed cleaner data into the feature engine
    df_features = fe.create_patient_features(df_spec_filtered, df_sick_filtered, df_treat_filtered)
    
    lens_cols, cluster_cols = fe.get_feature_sets(df_features)

    # 4. Run TDA Pipeline (GPU Accelerated)
    # Updated to use class-based TDAPipeline
    # Ensure output directories exist (Handled by Config, but safe to keep or rely on Config)
    # Config.setup_directories() is called on import
    
    mapper_path = os.path.join(Config.TEXT_OUTPUT_DIR, Config.FILE_MAPPER_HTML)
    pipeline = tp.TDAPipeline(df_features, lens_cols, cluster_cols, output_path=mapper_path)
    graph, ids = pipeline.run_mapper()
    
    # Step B: Analyze Topology (Loops & Super-Responders)
    super_responders, cycles = pipeline.analyze_topology()
    
    # [NEW] Step B-2: Save Static Image
    png_path = os.path.join(Config.PHOTO_OUTPUT_DIR, Config.FILE_TDA_GRAPH_IMG)
    pipeline.save_graph_as_png(png_path)

    # Step C: Export Results
    print("\n[4/4] Exporting Results...")
    
    # 1. Save Super-Responders
    print(f" - Found {len(super_responders)} Super-Responders.")
    if len(super_responders) > 0:
        out_path = os.path.join(Config.TEXT_OUTPUT_DIR, Config.FILE_SUPER_RESPONDERS)
        pd.DataFrame({'SPEC_ID_SNO': list(super_responders)}).to_csv(out_path, index=False)
        print(f"   -> Saved '{out_path}'")
        
    # 2. Extract and Save Loop Patients
    # cycles is a list of list of nodes (e.g. [[n1, n2, n3], [n4, n5...]])
    print(f" - Found {len(cycles)} Loops.")
    loop_patient_ids = set()
    
    if pipeline.graph is not None:
        for cycle_nodes in cycles:
            for node_id in cycle_nodes:
                if node_id in pipeline.graph['nodes']:
                    member_indices = pipeline.graph['nodes'][node_id]
                    # pipeline.df_pandas is available
                    pifs = pipeline.df_pandas.iloc[member_indices]['SPEC_ID_SNO'].tolist()
                    loop_patient_ids.update(pifs)
    
    print(f" - identified {len(loop_patient_ids)} unique patients in Loops.")
    if len(loop_patient_ids) > 0:
        out_path = os.path.join(Config.TEXT_OUTPUT_DIR, Config.FILE_LOOP_PATIENTS)
        pd.DataFrame({'SPEC_ID_SNO': list(loop_patient_ids)}).to_csv(out_path, index=False)
        print(f"   -> Saved '{out_path}'")

    print(f" - Check '{mapper_path}' and '{png_path}' for visualization.")
    print("="*60)
    print(" Analysis Complete.")
    print("="*60)

if __name__ == "__main__":
    main()
