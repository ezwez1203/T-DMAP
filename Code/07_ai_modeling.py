import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
import os
import importlib
import logging
from datetime import datetime
from tqdm import tqdm
from config import Config

# ==========================================
# GPU Acceleration Strategy (RAPIDS)
# ==========================================
USE_RAPIDS = False
try:
    import cudf
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    USE_RAPIDS = True and Config.USE_GPU
    if USE_RAPIDS:
        print("🚀 RAPIDS GPU Acceleration ENABLED")
except ImportError:
    print("⚠️  RAPIDS not found. Using CPU-based pandas/sklearn (slower)")
    cudf = pd  # Fallback to pandas

def _setup_logging() -> str:
    """Configure logging to both console and a txt file."""
    log_dir = Config.TEXT_OUTPUT_DIR
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(
        log_dir,
        f"ai_modeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Prevent duplicated handlers when re-running in interactive sessions.
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            try:
                if str(getattr(handler, "baseFilename", "")).startswith(log_dir) and str(
                    getattr(handler, "baseFilename", "")
                ).endswith(".txt"):
                    root.removeHandler(handler)
            except Exception:
                continue

    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_file_path


LOG_FILE_PATH = _setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {LOG_FILE_PATH}")

# Import Data Loader from existing module
dl = importlib.import_module("02_data_loader")

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() and Config.USE_GPU else 'cpu')
BATCH_SIZE = Config.LSTM_BATCH_SIZE
HIDDEN_DIM = Config.LSTM_HIDDEN_DIM
NUM_LAYERS = Config.LSTM_NUM_LAYERS
LEARNING_RATE = Config.LSTM_LEARNING_RATE
EPOCHS = Config.LSTM_EPOCHS

# ==========================================
# 1. Data Preparation
# ==========================================

def load_data():
    """
    Loads raw data and target labels.
    """
    logger.info("Loading Raw Data...")
    # Load using the existing loader
    spec, sick, treat = dl.load_and_merge_data(base_path=Config.DATA_DIR)
    
    # Convert Polars to Pandas for Modeling flexibility
    df_spec = spec.to_pandas()
    
    # Load Labels
    logger.info("Loading Labels (Loop vs Super-Responders)...")
    try:
        loop_path = os.path.join(Config.TEXT_OUTPUT_DIR, Config.FILE_LOOP_PATIENTS)
        resp_path = os.path.join(Config.TEXT_OUTPUT_DIR, Config.FILE_SUPER_RESPONDERS)
        
        loop_ids = pd.read_csv(loop_path)['SPEC_ID_SNO'].unique()
        resp_ids = pd.read_csv(resp_path)['SPEC_ID_SNO'].unique()
    except FileNotFoundError:
        logger.error("Critical CSVs not found. Please run '06_run_analysis.py' first.")
        raise
        
    # Create Label Metadata
    # Label: 1 = Loop (Bad), 0 = Super-Responder (Good)
    # Handle overlapping patients: prioritize Loop label (more critical)
    loop_set = set(loop_ids)
    resp_set = set(resp_ids) - loop_set  # Exclude overlapping patients from super-responders
    
    df_labels = pd.DataFrame({
        'SPEC_ID_SNO': list(loop_set) + list(resp_set),
        'label': [1]*len(loop_set) + [0]*len(resp_set)
    })
    
    logger.info(f"Label Stats - Loop: {len(loop_set)}, Super-Responders: {len(resp_set)}")
    
    # Filter Spec Data to only include these patients
    df_model = df_spec[df_spec['SPEC_ID_SNO'].isin(df_labels['SPEC_ID_SNO'])].copy()
    
    # Merge Labels
    df_model = df_model.merge(df_labels, on='SPEC_ID_SNO', how='inner')
    
    logger.info(f"Total Patients for Modeling: {df_model['SPEC_ID_SNO'].nunique()}")
    logger.info(f"Class Distribution:\n{df_model['label'].value_counts()}")
    
    return df_model

def preprocess_sequences(df):
    """
    Prepares sequential data for LSTM.
    Sorts by RECU_FR_DD (Care Start Date).
    """
    logger.info("Preprocessing Sequences...")
    
    # Ensure Date format
    if 'RECU_FR_DD' in df.columns:
        df['RECU_FR_DD'] = pd.to_datetime(df['RECU_FR_DD'], format='%Y-%m-%d', errors='coerce') # Format might vary
        # Fallback if standard format fails, try auto
        if df['RECU_FR_DD'].isnull().all():
             df['RECU_FR_DD'] = pd.to_datetime(df['RECU_FR_DD_RAW'], errors='coerce') # Hypothetical
             
        df = df.sort_values(by=['SPEC_ID_SNO', 'RECU_FR_DD'])
    else:
        logger.warning("'RECU_FR_DD' not found. Using original order (risk of temporal leakage).")
        
    # Feature Selection (Simple numeric features for demo)
    # In practice, one hot encode diagnoses, etc.
    # Assuming 'TOT_AMT' (Total Amount) exists. 
    # Attempting to find numeric columns
    feature_cols = []
    candidates = ['TOT_AMT', 'WRK_DYS_CNT', 'MDCT_DD_CNT', 'EDEC_ADD_RT', 'DRG_NO', 'V1', 'V2'] # Common HIRA fields
    
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and c != 'label' and c != 'SPEC_ID_SNO':
             # Simple heuristic: keep numeric columns with reasonable variance
             if df[c].nunique() > 1:
                 feature_cols.append(c)
                 
    # Limit features to top N for stability
    feature_cols = feature_cols[:10] 
    logger.info(f"Using Features for LSTM: {feature_cols}")
    
    # Fill NAs
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Normalize
    for c in feature_cols:
        mean = df[c].mean()
        std = df[c].std() + 1e-6
        df[c] = (df[c] - mean) / std
        
    # Group by Patient
    sequences = []
    labels = []
    
    grouped = df.groupby('SPEC_ID_SNO')
    for inputs, group in grouped:
        seq = group[feature_cols].values
        label = group['label'].iloc[0]
        sequences.append(torch.tensor(seq, dtype=torch.float32))
        labels.append(label)
        
    return sequences, labels, len(feature_cols)

# ==========================================
# 2. LSTM Model (Model A): Markov Chain-Based Loop Risk Prediction
# ==========================================
# 환자의 방문 이력이 마르코프 연쇄 속성을 가짐을 착안:
# - 현재 상태(질병, 처방)가 과거 이력에 의존
# - LSTM으로 장기 시퀀스에서 기울기 소실 문제 해결
# - h_t = σ(W · [h_{t-1}, x_t] + b)
# - 미래 루프 진입 확률을 예측

class VisitSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(yy, dtype=torch.float32).unsqueeze(1), torch.tensor(x_lens)

class RiskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RiskLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, lengths):
        # Pack
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (ht, ct) = self.lstm(packed_x)
        # Use last hidden state
        out = ht[-1] 
        return self.sigmoid(self.fc(out))

def train_lstm_model():
    logger.info("=== Training Model A: LSTM for Loop Risk ===")
    
    df = load_data()
    sequences, labels, input_dim = preprocess_sequences(df)
    
    # Check if we have both classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning(f"⚠️  Only one class found: {unique_labels}. Cannot train binary classifier properly.")
        logger.warning("Please check TDA analysis thresholds in config.py to generate both Loop and Super-Responder groups.")
        return
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Handle Class Imbalance with WeightedRandomSampler
    # Calculate weights
    y_train_indices = np.array(y_train)
    unique_train_labels = np.unique(y_train_indices)
    class_sample_count = np.array([len(np.where(y_train_indices == t)[0]) for t in unique_train_labels])
    weight = 1. / class_sample_count
    
    # Create weight mapping
    label_to_weight = {int(label): float(w) for label, w in zip(unique_train_labels, weight)}
    samples_weight = np.array([label_to_weight[int(y)] for y in y_train_indices])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_ds = VisitSequenceDataset(X_train, y_train)
    test_ds = VisitSequenceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
    
    model = RiskLSTM(input_dim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
        total_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}", unit="batch", leave=False)
        for step, (x, y, lens) in enumerate(batch_bar, start=1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x, lens)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

            avg_loss = total_loss / step
            batch_bar.set_postfix(loss=f"{avg_loss:.4f}")

        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")
            
    # Evaluate
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y, lens in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x, lens)
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    auc = roc_auc_score(all_targets, all_preds)
    logger.info("\nLSTM Test Results:")
    logger.info(f"AUC: {auc:.4f}")
    print(classification_report(all_targets, binary_preds))

# ==========================================
# 3. Escape Recommender (Model B)
# ==========================================

def train_escape_recommender():
    logger.info("=== Training Model B: Escape Recommender (Random Forest + SHAP) ===")
    
    df = load_data()
    
    # Check if we have both classes
    unique_labels = df['label'].unique()
    if len(unique_labels) < 2:
        logger.warning(f"⚠️  Only one class found in data: {unique_labels}. Cannot train RF classifier.")
        logger.warning("Please adjust TDA thresholds in config.py:")
        logger.warning("  - Increase TDA_HIGH_SEVERITY_Q (e.g., 0.75)")
        logger.warning("  - Decrease TDA_LOW_COST_Q (e.g., 0.25)")
        logger.warning("Then re-run 06_run_analysis.py to regenerate labels.")
        return
    
    # Static Feature Engineering (Aggregate)
    # We want to know WHAT feature makes them a Loop vs Super-Responder?
    # Drop non-feature cols
    drop_cols = ['SPEC_ID_SNO', 'label', 'RECU_FR_DD', 'KEY_SEQ']
    
    # Try to convert all to numeric, drop non-numeric
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # Retain the label and ID for grouping
    df_numeric['SPEC_ID_SNO'] = df['SPEC_ID_SNO']
    df_numeric['label'] = df['label']
    
    # Aggregate by Patient (Mean, Max, Sum)
    # Using simple Mean for now (interpretable for SHAP)
    # Keep label separate to avoid averaging
    df_agg = df_numeric.groupby('SPEC_ID_SNO').agg({
        **{col: 'mean' for col in df_numeric.columns if col not in ['SPEC_ID_SNO', 'label']},
        'label': 'first'  # Take the first label (they should all be the same per patient)
    })
    
    # Remove cols that are all NaN or constant
    df_agg = df_agg.dropna(axis=1, how='all')
    df_agg = df_agg.loc[:, (df_agg != df_agg.iloc[0]).any()] 
    
    # Ensure label exists before trying to drop
    if 'label' not in df_agg.columns:
        logger.error("Label column lost during aggregation. Cannot proceed.")
        return
    
    X = df_agg.drop(columns=['label'])
    y = df_agg['label']
    
    # Fill NAs
    X = X.fillna(0)
    
    # Check if we have enough variance in features
    if X.shape[1] == 0:
        logger.error("No features available after preprocessing. Cannot train model.")
        return
    
    logger.info(f"Training on {X.shape[0]} patients with {X.shape[1]} features")
    
    # Train Random Forest (with GPU acceleration if available)
    # Handle Class Imbalance: class_weight='balanced'
    if USE_RAPIDS:
        logger.info("Using cuML RandomForest (GPU-accelerated)")
        # Convert to cuDF for GPU processing
        X_gpu = cudf.DataFrame.from_pandas(X.reset_index(drop=True))
        y_gpu = cudf.Series(y.values)
        
        rf = cuRF(
            n_estimators=Config.RF_N_ESTIMATORS, 
            max_depth=Config.RF_MAX_DEPTH,
            random_state=Config.MODEL_SEED,
            n_bins=128,  # GPU optimization
            split_criterion=1,  # GINI
        )
        rf.fit(X_gpu, y_gpu)
        
        # Convert back to CPU for SHAP (SHAP doesn't support cuML yet)
        X_cpu = X
        y_cpu = y
    else:
        logger.info("Using sklearn RandomForest (CPU)")
        rf = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS, 
            max_depth=Config.RF_MAX_DEPTH, 
            class_weight=Config.RF_CLASS_WEIGHT, 
            random_state=Config.MODEL_SEED,
            n_jobs=-1  # Use all CPU cores
        )
        rf.fit(X, y)
        X_cpu = X
        y_cpu = y
    
    logger.info("Random Forest Training Complete.")
    
    # Evaluation
    if USE_RAPIDS:
        preds_gpu = rf.predict(X_gpu)
        probs_gpu = rf.predict_proba(X_gpu)
        
        # cuML returns cuDF Series/DataFrame, convert to numpy
        if hasattr(preds_gpu, 'to_numpy'):
            preds = preds_gpu.to_numpy()
        else:
            preds = preds_gpu.values_host  # Alternative cuML method
        
        # Handle predict_proba output (could be cuDF DataFrame)
        if hasattr(probs_gpu, 'to_numpy'):
            probs_array = probs_gpu.to_numpy()
        else:
            probs_array = probs_gpu.as_matrix() if hasattr(probs_gpu, 'as_matrix') else probs_gpu
        
        # Get class 1 probabilities
        if probs_array.ndim == 2:
            probs = probs_array[:, 1] if probs_array.shape[1] > 1 else probs_array[:, 0]
        else:
            probs = probs_array
    else:
        preds = rf.predict(X)
        probs = rf.predict_proba(X)[:, 1]
    
    logger.info("\nRandom Forest Results (Training Set):")
    print(classification_report(y_cpu, preds))
    
    # SHAP Explainability (CPU only - SHAP doesn't support cuML)
    logger.info("Generating SHAP Explanations...")
    
    # For RAPIDS models, need to convert to sklearn-compatible format
    if USE_RAPIDS:
        logger.info("Converting cuML model to CPU for SHAP analysis...")
        # Re-train a small sklearn model for SHAP (explainability trade-off)
        rf_shap = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=Config.RF_MAX_DEPTH, 
            random_state=Config.MODEL_SEED
        )
        rf_shap.fit(X_cpu, y_cpu)
        explainer = shap.TreeExplainer(rf_shap)
    else:
        explainer = shap.TreeExplainer(rf)
    
    shap_values = explainer.shap_values(X_cpu)
    
    # Handling Binary Classification SHAP output format (list of 2 arrays, take index 1 for class 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_cpu, show=False)
    
    output_path = os.path.join(Config.PHOTO_OUTPUT_DIR, Config.FILE_SHAP_IMG)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP Summary Plot saved to {output_path}")

def main():
    print("=" * 80)
    print("AI Modeling Pipeline for Chronic Loop Risk & Escape Factors")
    print("=" * 80)
    print("📊 Hardware Configuration:")
    print(f"   - GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   - RAPIDS Acceleration: {USE_RAPIDS}")
    print(f"   - Device: {DEVICE}")
    print("=" * 80)
    
    # Ensure dependencies
    loop_path = os.path.join(Config.TEXT_OUTPUT_DIR, Config.FILE_LOOP_PATIENTS)
    if not os.path.exists(loop_path):
        print(f"\n❌ Error: '{loop_path}' missing.")
        print("📋 Action Required:")
        print("   1. Check config.py TDA thresholds:")
        print("      - TDA_HIGH_SEVERITY_Q = 0.75 (stricter)")
        print("      - TDA_LOW_COST_Q = 0.25 (stricter)")
        print("   2. Run: python3 06_run_analysis.py")
        print("   3. Then re-run this script")
        return

    # Run Model A: LSTM (Markov Chain-based Sequence Model)
    print("\n🔄 Starting Model A: LSTM Loop Risk Predictor...")
    train_lstm_model()
    
    # Run Model B: Random Forest + SHAP (Escape Factor Recommender)
    print("\n🔄 Starting Model B: Escape Factor Recommender...")
    train_escape_recommender()
    
    print("\n" + "=" * 80)
    print("✅ AI Modeling Pipeline Complete")
    print(f"📁 Results saved to: {Config.OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
