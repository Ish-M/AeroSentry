import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import socket
import pandas as pd
import numpy as np
import torch
from collections import deque
from models import AeroSentryLSTM

HOST = '127.0.0.1'
PORT = 9999
SEQUENCE_LENGTH = 30
MODEL_WEIGHTS_PATH = 'models/aerosentry_v1.pth'

CMAPSS_COLUMNS = [
    'unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'
] + [f'sensor_measurement_{i}' for i in range(1, 22)]

FEATURES_TO_USE = [col for col in CMAPSS_COLUMNS if col not in [
    'setting_3', 'sensor_measurement_1', 'sensor_measurement_5', 
    'sensor_measurement_10', 'sensor_measurement_16', 'sensor_measurement_18', 'sensor_measurement_19'
]]

SCALING_FEATURES = [col for col in FEATURES_TO_USE if col not in ['unit_number', 'time_in_cycles']]

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

def calculate_cmapss_score(d):
    """Official NASA asymmetric scoring penalty function."""
    if d < 0:
        return np.exp(-d / 10.0) - 1
    else:
        return np.exp(d / 13.0) - 1

def generate_summary_report(session_records):
    print("\n" + "="*80)
    print(f"{'AEROSENTRY OFFICIAL NASA BENCHMARK DIAGNOSTIC REPORT':^80}")
    print("="*80)
    
    if not session_records:
        print("[-] No operational windows were processed. Summary unavailable.")
        print("="*80)
        return

    df_records = pd.DataFrame(session_records)
    
    print(f"{'Engine Unit':<12} | {'Total Cycles':<14} | {'First Failure Alert':<20} | {'Final Point NASA Score'}")
    print("-"*80)
    
    total_fleet_score = 0.0
    
    for unit, group in df_records.groupby('unit'):
        group = group.sort_values('cycle')
        max_cycle = group['cycle'].max()
        final_row = group.iloc[-1]
        
        failing_alerts = group[group['status'] == "🔴 FAILING DETECTED"]
        first_alert_cycle = failing_alerts['cycle'].min() if not failing_alerts.empty else "N/A"
        
        # NASA Benchmark standard: Evaluate error 'd' ONLY at the final operational cutoff cycle
        # True RUL at the end of the data file is exactly 0
        true_final_rul = 0 
        estimated_final_rul = final_row['pred_rul']
        
        d = estimated_final_rul - true_final_rul
        unit_final_score = calculate_cmapss_score(d)
        total_fleet_score += unit_final_score
        
        alert_str = f"Cycle {first_alert_cycle}" if first_alert_cycle != "N/A" else "Never Flagged"
        print(f"Engine #{unit:<5} | {max_cycle:<14} | {alert_str:<20} | {unit_final_score:.4f}")
    
    print("="*80)
    print(f"• Total Fleet Units Monitored (n)  : {len(df_records['unit'].unique())}")
    print(f"• Final Cumulative NASA Fleet Score: {total_fleet_score:.4f}")
    print("="*80 + "\n")

def launch_inference_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AeroSentryLSTM(sequence_length=SEQUENCE_LENGTH, num_features=len(SCALING_FEATURES))
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"[+] Loaded trained model weights.")
    except FileNotFoundError:
        print(f"[-] Error: Weights missing at {MODEL_WEIGHTS_PATH}.")
        return

    pipeline_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        pipeline_socket.connect((HOST, PORT))
        print("[+] Linked to telemetry feed stream loop.")
    except ConnectionRefusedError:
        print("[-] Connection Refused. Start mock_streamer.py first.")
        return

    session_records = []
    history_window = deque(maxlen=SEQUENCE_LENGTH)
    buffer = ""
    
    # State tracking variable to detect engine transitions
    current_active_unit = None

    print("\n" + "="*70)
    print(f"{'AEROSENTRY LIVE PREDICTIVE MAINTENANCE DASHBOARD':^70}")
    print("="*70)
    print(f"{'Unit':<6} | {'Cycle':<6} | {'System Operational Status':<22} | {'Predicted RUL':<15}")
    print("-"*70)

    try:
        while True:
            data = pipeline_socket.recv(4096).decode('utf-8')
            if not data:
                break
            
            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue
                
                raw_values = [float(x) for x in line.split() if x.strip()]
                if len(raw_values) == len(CMAPSS_COLUMNS):
                    df_row = pd.DataFrame([raw_values], columns=CMAPSS_COLUMNS)
                    unit = int(df_row['unit_number'].iloc[0])
                    cycle = int(df_row['time_in_cycles'].iloc[0])
                    
                    # MEMORY FIX: If a new engine appears, completely clear out the window queue
                    if current_active_unit is not None and unit != current_active_unit:
                        history_window.clear()
                    current_active_unit = unit
                    
                    # Apply dynamic Z-score tracking array transformation
                    normalized_vector = []
                    for col in SCALING_FEATURES:
                        val = df_row[col].iloc[0]
                        mean = FLEET_STATS[col]['mean']
                        std = FLEET_STATS[col]['std']
                        val = (val - mean) / std
                        normalized_vector.append(val)
                        
                    history_window.append(normalized_vector)
                    
                    if len(history_window) == SEQUENCE_LENGTH:
                        input_tensor = torch.tensor(list(history_window), dtype=torch.float32).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            predicted_rul = max(0, int(model(input_tensor).item()))
                        
                        # Piecewise-scaled real-time flag thresholds
                        if predicted_rul <= 15:
                            status = "🔴 FAILING DETECTED"
                        elif predicted_rul <= 45:
                            status = "⚠️ DEGRADATION WARNING"
                        else:
                            status = "✅ NOMINAL HEALTH"
                            
                        print(f"{unit:<6} | {cycle:<6} | {status:<22} | {predicted_rul:<12} cycles")
                        
                        session_records.append({
                            'unit': unit,
                            'cycle': cycle,
                            'pred_rul': predicted_rul,
                            'status': status
                        })
                    else:
                        print(f"{unit:<6} | {cycle:<6} | ⏳ INITIALIZING WINDOW      | ({len(history_window)}/{SEQUENCE_LENGTH})")
                        
    except KeyboardInterrupt:
        print("\n[-] Ingestion halted via manual keyboard override.")
    finally:
        pipeline_socket.close()
        generate_summary_report(session_records)

if __name__ == '__main__':
    launch_inference_pipeline()