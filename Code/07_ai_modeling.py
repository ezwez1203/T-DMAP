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
from tqdm import tqdm
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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
    df_labels = pd.DataFrame({
        'SPEC_ID_SNO': np.concatenate([loop_ids, resp_ids]),
        'label': [1]*len(loop_ids) + [0]*len(resp_ids)
    })
    
    # Remove duplicates (if any patient is somehow in both, though unlikely in TDA logic usually)
    df_labels = df_labels.drop_duplicates(subset=['SPEC_ID_SNO'])
    
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
# 2. LSTM Model (Model A)
# ==========================================

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
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Handle Class Imbalance with WeightedRandomSampler
    # Calculate weights
    y_train_indices = [y for y in y_train]
    class_sample_count = np.array([len(np.where(y_train_indices == t)[0]) for t in np.unique(y_train_indices)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train_indices])
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
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y, lens in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x, lens)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
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
    df_agg = df_numeric.groupby('SPEC_ID_SNO').mean()
    
    # Remove cols that are all NaN or constant
    df_agg = df_agg.dropna(axis=1, how='all')
    df_agg = df_agg.loc[:, (df_agg != df_agg.iloc[0]).any()] 
    
    X = df_agg.drop(columns=['label'])
    y = df_agg['label']
    
    # Fill NAs
    X = X.fillna(0)
    
    # Train Random Forest
    # Handle Class Imbalance: class_weight='balanced'
    rf = RandomForestClassifier(n_estimators=Config.RF_N_ESTIMATORS, 
                                max_depth=Config.RF_MAX_DEPTH, 
                                class_weight=Config.RF_CLASS_WEIGHT, 
                                random_state=Config.MODEL_SEED)
    rf.fit(X, y)
    
    logger.info("Random Forest Training Complete.")
    
    # Evaluation
    preds = rf.predict(X)
    probs = rf.predict_proba(X)[:, 1]
    logger.info("\nRandom Forest Results (Training Set):")
    print(classification_report(y, preds))
    
    # SHAP Explainability
    logger.info("Generating SHAP Explanations...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    
    # Handling Binary Classification SHAP output format (list of 2 arrays, take index 1 for class 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    
    output_path = os.path.join(Config.PHOTO_OUTPUT_DIR, Config.FILE_SHAP_IMG)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP Summary Plot saved to {output_path}")

def main():
    print("AI Modeling Pipeline for Chronic Loop Risk & Escape Factors")
    
    # Ensure dependencies
    loop_path = os.path.join(Config.TEXT_OUTPUT_DIR, Config.FILE_LOOP_PATIENTS)
    if not os.path.exists(loop_path):
        print(f"Error: '{loop_path}' missing. Run 06_run_analysis.py first.")
        return

    # Run Model A
    train_lstm_model()
    
    # Run Model B
    train_escape_recommender()

if __name__ == "__main__":
    main()
