import polars as pl

from config import Config

class DataFilter:
    """
    Module for cleaning and filtering raw HIRA data before Feature Engineering.
    Focuses on removing noise, handling missing values, and selecting relevant features.
    """
    def __init__(self, df_merged, df_sick):
        self.df = df_merged
        self.df_sick = df_sick

    def apply_filters(self):
        print("[1.5/4] Filtering Data (Noise Reduction & Feature Selection)...")
        
        # 1. Column Selection (Dimensionality Reduction)
        # We only keep columns relevant to 'Severity' (Input) and 'Outcome' (Output).
        # Dropping administrative columns like 'JID', 'DGSBJT_CD', etc.
        target_cols = [
            "SPEC_ID_SNO",  # Key
            "age_group",    # Demographic
            "MSICK_CD",     # Clinical Input (Diagnosis)
            "AMT",          # Outcome (Cost)
            "TOT_USE_QTY",  # Outcome (Intensity)
            "GNL_NM_CD"     # Clinical Input (Treatment)
        ]
        
        # Check if columns exist before selecting to avoid errors
        existing_cols = [col for col in target_cols if col in self.df.columns]
        self.df = self.df.select(existing_cols)
        
        initial_rows = len(self.df)

        # 2. Data Cleaning (Row Filtering)
        # Remove rows with missing IDs (Orphan data)
        self.df = self.df.filter(pl.col("SPEC_ID_SNO").is_not_null())
        
        # Filter Invalid Costs (Negative amounts are usually billing corrections)
        if "AMT" in self.df.columns:
            self.df = self.df.filter(pl.col("AMT") >= 0)

        # 3. Outlier Handling (Clipping)
        # Extreme costs (e.g., top 0.1%) can distort the TDA shape.
        # We cap the AMT at the 99th percentile.
        if "AMT" in self.df.columns:
            # Calculate 99th percentile using approx_quantile
            quantile_val = Config.OUTLIER_QUANTILE
            quantile_limit = self.df.select(
                pl.col("AMT").quantile(quantile_val)
            ).to_numpy()[0][0]
            
            # Cap values greater than 99th percentile
            if quantile_limit is not None:
                self.df = self.df.with_columns(
                    pl.when(pl.col("AMT") > quantile_limit)
                    .then(quantile_limit)
                    .otherwise(pl.col("AMT"))
                    .alias("AMT")
                )
                print(f"   - Outlier Clipping: Capped AMT at {quantile_limit:.2f} (Quantile {quantile_val})")

        # 4. Filter Diagnosis Data (df_sick)
        # Remove empty diagnosis codes
        self.df_sick = self.df_sick.filter(
            pl.col("SICK_CD").is_not_null() & (pl.col("SICK_CD") != "")
        )

        final_rows = len(self.df)
        print(f"   - Rows processed: {initial_rows} -> {final_rows} (Removed {initial_rows - final_rows} noisy rows)")
        
        return self.df, self.df_sick

if __name__ == "__main__":
    # Test execution
    print("\n[Testing DataFilter Module]")
    
    # Create dummy data
    df_merged = pl.DataFrame({
        "SPEC_ID_SNO": [1, 2, 3, None, 5],
        "age_group": [20, 30, 40, 50, 60],
        "MSICK_CD": ["A", "B", "C", "D", "E"],
        "AMT": [1000, 2000, -500, 3000, 1000000],  # Includes negative and outlier
        "TOT_USE_QTY": [10, 20, 30, 40, 50],
        "GNL_NM_CD": ["D1", "D2", "D3", "D4", "D5"]
    })
    
    df_sick = pl.DataFrame({
        "SPEC_ID_SNO": [1, 2, 3, 5],
        "SICK_CD": ["A01", None, "B02", ""] # Includes None and empty
    })

    print(f"Original Data Shape: {df_merged.shape}")
    
    filter = DataFilter(df_merged, df_sick)
    df_m_filtered, df_s_filtered = filter.apply_filters()
    
    print(f"Filtered Data Shape: {df_m_filtered.shape}")
    print(df_m_filtered)
    print("Test Complete.")
