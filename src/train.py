import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from models import AeroSentryLSTM

# Optimized Hyperparameters for Deep LSTM Convergence
SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 25 
LEARNING_RATE = 0.001
DATA_PATH = os.path.join('data', 'CMAPPS', 'train_FD001.txt')
MODEL_SAVE_PATH = os.path.join('models', 'aerosentry_v1.pth')

CMAPSS_COLUMNS = [
    'unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'
] + [f'sensor_measurement_{i}' for i in range(1, 22)]

# Drop uninformative/flatline sensors
FEATURES_TO_USE = [col for col in CMAPSS_COLUMNS if col not in [
    'setting_3', 'sensor_measurement_1', 'sensor_measurement_5', 
    'sensor_measurement_10', 'sensor_measurement_16', 'sensor_measurement_18', 'sensor_measurement_19'
]]

# Isolate just the actual numeric features to normalize (dropping IDs and time)
SCALING_FEATURES = [col for col in FEATURES_TO_USE if col not in ['unit_number', 'time_in_cycles']]

# Hardcoded global distribution stats for FD001 training data stabilization
FLEET_STATS = {
    'setting_1': {'mean': -0.000004, 'std': 0.0021},
    'setting_2': {'mean': 0.0002, 'std': 0.0003},
    'sensor_measurement_2': {'mean': 642.6809, 'std': 0.5003},
    'sensor_measurement_3': {'mean': 1590.5231, 'std': 6.1311},
    'sensor_measurement_4': {'mean': 1408.9338, 'std': 9.0006},
    'sensor_measurement_6': {'mean': 21.6098, 'std': 0.0014},
    'sensor_measurement_7': {'mean': 553.3677, 'std': 0.8850},
    'sensor_measurement_8': {'mean': 2388.0966, 'std': 0.0710},
    'sensor_measurement_9': {'mean': 9065.2429, 'std': 22.0828},
    'sensor_measurement_11': {'mean': 47.5412, 'std': 0.2671},
    'sensor_measurement_12': {'mean': 521.4134, 'std': 0.7375},
    'sensor_measurement_13': {'mean': 2388.0962, 'std': 0.0719},
    'sensor_measurement_14': {'mean': 8143.7514, 'std': 19.0761},
    'sensor_measurement_15': {'mean': 8.4421, 'std': 0.0375},
    'sensor_measurement_17': {'mean': 393.2106, 'std': 1.5487},
    'sensor_measurement_20': {'mean': 38.8163, 'std': 0.1807},
    'sensor_measurement_21': {'mean': 23.2897, 'std': 0.1083}
}

class CMAPSSDataset(Dataset):
    def __init__(self, data_path, seq_length=30):
        df = pd.read_csv(data_path, sep=r'\s+', names=CMAPSS_COLUMNS)
        
        # PIECEWISE RUL: Cap ground-truth target values to 125 cycles max
        max_cycles = df.groupby('unit_number')['time_in_cycles'].max().to_dict()
        linear_rul = df['unit_number'].map(max_cycles) - df['time_in_cycles']
        df['RUL'] = np.minimum(linear_rul, 125)
        
        # Synchronized Z-score Standardization
        for col in SCALING_FEATURES:
            mean = FLEET_STATS[col]['mean']
            std = FLEET_STATS[col]['std']
            df[col] = (df[col] - mean) / std

        self.sequences = []
        self.labels = []
        
        for unit in df['unit_number'].unique():
            unit_df = df[df['unit_number'] == unit].sort_values('time_in_cycles')
            num_rows = len(unit_df)
            
            if num_rows >= seq_length:
                # We drop unit_number and time_in_cycles from the model inputs
                feature_array = unit_df[SCALING_FEATURES].values
                rul_array = unit_df['RUL'].values
                
                for i in range(num_rows - seq_length + 1):
                    self.sequences.append(feature_array[i : i + seq_length])
                    self.labels.append(rul_array[i + seq_length - 1])
                    
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])

def run_training():
    os.makedirs('models', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Initializing LSTM training loop using engine hardware: {device}")
    
    dataset = CMAPSSDataset(DATA_PATH, seq_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Passing 17 features down to match SCALING_FEATURES length
    model = AeroSentryLSTM(sequence_length=SEQUENCE_LENGTH, num_features=len(SCALING_FEATURES)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"[+] Processing {len(dataset)} sequence arrays. Starting optimization...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        total_epoch_mse = epoch_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Average Training MSE Loss: {total_epoch_mse:.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[+] Exported trained LSTM weights matrix securely to: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    run_training()