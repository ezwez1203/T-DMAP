
import polars as pl
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import Config

def load_and_merge_data(base_path=Config.DATA_DIR):
    """
    Loads EDU200, EDU400, EDU530 from the specified directory.
    Renames 'MID' to 'SPEC_ID_SNO' if necessary.
    Merges them into a single Dataframe for analysis.
    """
    try:
        # 1. Load Data
        logger.info(f"Loading CSV files from {base_path}...")
        
        # Helper to load file directly
        def load_file(filename):
            path = f"{base_path}/{filename}"
            try:
                return pl.read_csv(path, infer_schema_length=10000, ignore_errors=True)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                raise

        # ST200: Specification (General Info)
        df_spec = load_file(Config.FILE_SPEC)
        # ST400: Sickness/Diagnosis
        df_sick = load_file(Config.FILE_SICK)
        # ST530: Treatment/Prescription
        df_treat = load_file(Config.FILE_TREAT)
        
        # 2. Rename Keys (User Requirement: Explicit join key SPEC_ID_SNO)
        def standardize_columns(df, type_name):
            if "MID" in df.columns:
                logger.info(f"Renaming 'MID' to 'SPEC_ID_SNO' in {type_name}")
                return df.rename({"MID": "SPEC_ID_SNO"})
            elif "명세서조인키" in df.columns: # Fallback for Korean
                logger.info(f"Renaming '명세서조인키' to 'SPEC_ID_SNO' in {type_name}")
                return df.rename({"명세서조인키": "SPEC_ID_SNO"})
            return df

        df_spec = standardize_columns(df_spec, "ST200")
        df_sick = standardize_columns(df_sick, "ST400")
        df_treat = standardize_columns(df_treat, "ST530")

        # Basic Check
        if "SPEC_ID_SNO" not in df_spec.columns:
            # If empty (fail to load), this might trigger
            if df_spec.is_empty():
                 logger.warning("ST200/EDU200 is empty or not found.")
            else:
                 raise ValueError("Critical: 'SPEC_ID_SNO' column missing after renaming attempts.")

        # 3. Aggregations for Feature Engineering Preparation
        unique_patients = df_spec.select("SPEC_ID_SNO").unique().height
        logger.info(f"Loaded {unique_patients} unique patients from EDU200.")
        
        return df_spec, df_sick, df_treat

    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise

if __name__ == "__main__":
    spec, sick, treat = load_and_merge_data()
    print(spec.head())
