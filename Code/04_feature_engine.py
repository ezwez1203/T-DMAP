
import polars as pl
import logging

from config import Config

logger = logging.getLogger(__name__)

def create_patient_features(df_spec, df_sick, df_treat):
    """
    Transforms raw transactional data into patient-level feature vectors.
    """
    logger.info("Starting Feature Engineering...")
    
    # 1. Lens Features (Global Shape)
    # Comorbidity Count
    comorbidity = df_sick.group_by("SPEC_ID_SNO").agg([
        pl.col("SICK_CD").n_unique().alias("Comorbidity_Count")
    ])
    
    # Cost & Duration (Visit Count)
    treatment_agg = df_treat.group_by("SPEC_ID_SNO").agg([
        pl.col("AMT").sum().fill_null(0).log1p().alias("Total_Cost"), 
        pl.col("TOT_USE_QTY").sum().fill_null(0).log1p().alias("Total_Med_Qty"), # Updated col name based on rename mapping if applicable
        pl.len().alias("Visit_Count")
    ])
    
    # 2. Clustering Features (Local Detail - Drug Vectors)
    # We take the Top N most common drugs to keep dimensionality manageable for DBSCAN
    if "GNL_NM_CD" in df_treat.columns:
        top_drugs = (df_treat.group_by("GNL_NM_CD")
                     .count()
                     .sort("count", descending=True)
                     .head(Config.TOP_DRUG_COUNT)
                     .select("GNL_NM_CD"))
        
        # Filter treatment data to only these drugs
        df_treat_filtered = df_treat.join(top_drugs, on="GNL_NM_CD", how="inner")
        
        # Pivot to create Patient x Drug Matrix (Count)
        drug_vectors = (df_treat_filtered
                        .group_by(["SPEC_ID_SNO", "GNL_NM_CD"])
                        .count()
                        .pivot(index="SPEC_ID_SNO", columns="GNL_NM_CD", values="count")
                        .fill_null(0))
        
        # Prefix columns to avoid collision
        drug_cols = [c for c in drug_vectors.columns if c != "SPEC_ID_SNO"]
        drug_vectors = drug_vectors.rename({c: f"Drug_{c}" for c in drug_cols})
    else:
        logger.warning("GNL_NM_CD column not found. Skipping Drug Vectors.")
        drug_vectors = df_treat.select("SPEC_ID_SNO").unique()
    
    # 3. Merge All
    df_features = df_spec.select(["SPEC_ID_SNO"])
    
    df_features = df_features.join(comorbidity, on="SPEC_ID_SNO", how="left").fill_null(0)
    df_features = df_features.join(treatment_agg, on="SPEC_ID_SNO", how="left").fill_null(0)
    if "GNL_NM_CD" in df_treat.columns:
        df_features = df_features.join(drug_vectors, on="SPEC_ID_SNO", how="left").fill_null(0)
    
    logger.info(f"Feature Engineering Complete. Shape: {df_features.shape}")
    return df_features

def get_feature_sets(df_features):
    """
    Returns separate feature sets for Lens and Clustering as requested.
    """
    # Lens (Global Shape): Cost, Duration, Comorbidity
    lens_cols = ["Total_Cost", "Visit_Count", "Comorbidity_Count"]
    
    # Clustering (Local Detail): Drug Vectors
    # Select all columns starting with "Drug_"
    cluster_cols = [c for c in df_features.columns if c.startswith("Drug_")]
    
    # If no drugs found (e.g. empty data), fallback to Med Qty
    if not cluster_cols:
        logger.warning("No drug vectors found. Using generic medication quantity.")
        existing = [c for c in ["Total_Med_Qty"] if c in df_features.columns]
        cluster_cols = existing
        
    return lens_cols, cluster_cols

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n[Testing FeatureEngine Module]")
    
    # Dummy Data
    df_spec = pl.DataFrame({"SPEC_ID_SNO": [1, 2, 3]})
    df_sick = pl.DataFrame({
        "SPEC_ID_SNO": [1, 1, 2, 3],
        "SICK_CD": ["A", "B", "A", "C"]
    })
    df_treat = pl.DataFrame({
        "SPEC_ID_SNO": [1, 1, 2, 2, 3],
        "AMT": [100, 200, 150, 150, 300],
        "TOT_USE_QTY": [1, 1, 2, 1, 5],
        "GNL_NM_CD": ["D1", "D2", "D1", "D3", "D1"]
    })
    
    features = create_patient_features(df_spec, df_sick, df_treat)
    print(f"Features created: {features.shape}")
    print(features)
    
    lens, cluster = get_feature_sets(features)
    print(f"Lens Columns: {lens}")
    print(f"Cluster Columns: {cluster}")
    print("Test Complete.")
