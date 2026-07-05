import os
# 1. Clear the OpenMP collision right at startup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import pandas as pd
import numpy as np
import torch
from models import AeroSentryLSTM

# 2. Declare Data Layout Configurations
CMAPSS_COLUMNS = [
    'unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'
] + [f'sensor_measurement_{i}' for i in range(1, 22)]

FEATURES_TO_USE = [col for col in CMAPSS_COLUMNS if col not in [
    'setting_3', 'sensor_measurement_1', 'sensor_measurement_5', 
    'sensor_measurement_10', 'sensor_measurement_16', 'sensor_measurement_18', 'sensor_measurement_19'
]]

SCALING_FEATURES = [col for col in FEATURES_TO_USE if col not in ['unit_number', 'time_in_cycles']]

# 3. Declare Hardcoded Baseline Z-Score Distribution Parameters
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

# 4. Declare Mathematical Scoring Function
def calculate_cmapss_score(d):
    if d < 0:
        return np.exp(-d / 10.0) - 1
    else:
        return np.exp(d / 13.0) - 1

def run_clean_batch_benchmark():
    data_path = os.path.join('data', 'CMAPPS', 'train_FD001.txt')
    model_path = os.path.join('models', 'aerosentry_v1.pth')
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("[-] Error: Missing required training data or model weights file.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize network model structure
    model = AeroSentryLSTM(sequence_length=30, num_features=len(SCALING_FEATURES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load raw text matrix
    df = pd.read_csv(data_path, sep=r'\s+', names=CMAPSS_COLUMNS)
    
    total_fleet_score = 0.0
    total_logs_processed = 0
    critical_alerts = 0
    engine_scores = []
    
    start_time = time.time()
    
    # 5. Execute boundary-safe grouping loop
    for unit, group in df.groupby('unit_number'):
        group = group.sort_values('time_in_cycles')
        max_cycle = group['time_in_cycles'].max()
        
        # Apply Z-score transformation safely within this isolated slice
        unit_features = group.copy()
        for col in SCALING_FEATURES:
            mean = FLEET_STATS[col]['mean']
            std = FLEET_STATS[col]['std']
            unit_features[col] = (unit_features[col] - mean) / std
            
        normalized_values = unit_features[SCALING_FEATURES].values
        
        if len(normalized_values) >= 30:
            # Isolate the final 30-cycle operational window context
            final_window = normalized_values[-30:]
            input_tensor = torch.tensor(final_window, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_rul = max(0, int(model(input_tensor).item()))
            
            # Count historical statistics for checking
            total_logs_processed += len(normalized_values) - 30 + 1
            if pred_rul <= 15:
                critical_alerts += 1
                
            # True final RUL at terminal cutoff point is exactly 0
            d = pred_rul - 0 
            unit_score = calculate_cmapss_score(d)
            total_fleet_score += unit_score
            
            engine_scores.append({
                'unit': unit,
                'max_cycle': max_cycle,
                'pred_rul': pred_rul,
                'score': unit_score
            })
            
    execution_time = time.time() - start_time
    
    # 6. Print Formatted Dashboard Report Interface
    print("\n" + "="*50)
    print(f"{'HIGH-SPEED BOUNDARY BENCHMARK COMPLETE':^50}")
    print("="*50)
    print(f" Execution Time             : {execution_time:.4f} seconds")
    print(f" Total Engine Logs Evaluated: {total_logs_processed:,}")
    print(f" Total Assets Evaluated (n) : {len(engine_scores)}")
    print(f" True Cumulative NASA Score : {total_fleet_score:.4f}")
    print("="*50)
    
    # Sort and slice to look at your top 5 highest penalties
    df_scores = pd.DataFrame(engine_scores).sort_values('score', ascending=False)
    print(" Top 5 Engines with Highest Penalty Scores:")
    for _, row in df_scores.head(5).iterrows():
        print(f"  ⚙️ Engine_{int(row['unit']):03d} -> Final Pred RUL: {int(row['pred_rul']):<3} | Penalty: {row['score']:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    run_clean_batch_benchmark()