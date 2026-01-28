import os

class Config:
    # ==========================================
    # 1. System & Paths
    # ==========================================
    # Hardware
    USE_GPU = True  # Set to False to force CPU for TDA/TensorFlow/PyTorch if supported
    
    # Base Directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "csv", "standardized_data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "result")
    
    # Sub-directories for outputs
    TEXT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "text")
    PHOTO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "photo")
    
    # Input File Names (Relative to DATA_DIR)
    FILE_SPEC = "ST200_renamed.csv"   # Specification / Demographics
    FILE_SICK = "ST400_renamed.csv"   # Diagnosis
    FILE_TREAT = "ST530_renamed.csv"  # Treatment / Prescription
    
    # Output File Names (Relative to TEXT_OUTPUT_DIR)
    FILE_SUPER_RESPONDERS = "super_responders_ids.csv"
    FILE_LOOP_PATIENTS = "loop_patients_ids.csv"
    FILE_MAPPER_HTML = "mapper_output.html"
    
    # Output Images (Relative to PHOTO_OUTPUT_DIR)
    FILE_TDA_GRAPH_IMG = "tda_disease_map.png"
    FILE_SHAP_IMG = "solution_shap.png"

    # ==========================================
    # 2. Data Filtering & Feature Engineering
    # ==========================================
    OUTLIER_QUANTILE = 0.99  # Cap AMT at this percentile
    TOP_DRUG_COUNT = 50      # Number of top drugs to keep for clustering features
    
    # ==========================================
    # 3. TDA Pipeline (KeplerMapper)
    # ==========================================
    # Lens / Projection (UMAP)
    TDA_METRIC = 'euclidean'
    TDA_N_NEIGHBORS = 30
    TDA_MIN_DIST = 0.1
    TDA_N_COMPONENTS = 2
    TDA_RANDOM_STATE = 42
    
    # Cover (Mapper)
    TDA_RESOLUTION = 15      # n_cubes (Resolution)
    TDA_OVERLAP = 0.5        # perc_overlap
    
    # Clustering (DBSCAN)
    TDA_EPS = 1.5            # Epsilon neighborhood for DBSCAN
    TDA_MIN_SAMPLES = 2      # Min points to form a cluster
    
    # [수정 포인트!] Analysis Thresholds (양극단 분리 전략)
    # ---------------------------------------------------------
    # 목표: Loop와 Super-Responder를 균형있게 분리
    # Loop: 상위 60% 이상 심각 (quantile 0.4 이상)
    # Super-Responder: 하위 40% 비용 (quantile 0.4 이하)
    TDA_HIGH_SEVERITY_Q = 0.40  # 상위 60% 심각도를 Loop로 인정
    
    # Super-Responder: 하위 40% 비용까지 인정 (완화)
    TDA_LOW_COST_Q = 0.40
    # ---------------------------------------------------------

    # ==========================================
    # 4. AI Modeling
    # ==========================================
    # General
    MODEL_SEED = 42
    TEST_SIZE = 0.2
    
    # GPU Memory Optimization (for 16GB VRAM)
    # - Batch processing for large datasets
    # - Data type optimization (float32 instead of float64)
    GPU_BATCH_SIZE = 10000  # Process 10K patients at a time
    USE_FLOAT32 = True      # Memory optimization
    
    # Model A: LSTM (Chronic Loop Prediction)
    # - Markov Chain-based sequential modeling
    # - Predicts loop entry probability from visit history
    LSTM_HIDDEN_DIM = 64
    LSTM_NUM_LAYERS = 2
    LSTM_LEARNING_RATE = 0.001
    LSTM_EPOCHS = 20
    LSTM_BATCH_SIZE = 32
    
    # Model B: Random Forest (Escape Factor)
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 5
    RF_CLASS_WEIGHT = 'balanced'

    @classmethod
    def setup_directories(cls):
        """Ensures output directories exist."""
        os.makedirs(cls.TEXT_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.PHOTO_OUTPUT_DIR, exist_ok=True)

# Run setup on import to ensure folders exist
Config.setup_directories()