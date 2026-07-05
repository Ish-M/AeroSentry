import socket
import pandas as pd
import numpy as np

HOST = '127.0.0.1'
PORT = 9999

# Standardized structural configuration matrix for the 26 C-MAPSS data features
CMAPSS_COLUMNS = [
    'unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'
] + [f'sensor_measurement_{i}' for i in range(1, 22)]

def launch_pipeline():
    pipeline_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        pipeline_socket.connect((HOST, PORT))
        print("[+] Ingestion Pipeline linked to Mock Telemetry Streamer successfully.")
    except ConnectionRefusedError:
        print("[-] Connection Refused: Is mock_streamer.py actively running in your other terminal panel?")
        return

    buffer = ""
    print("\n" + "="*65)
    print(f"{'AEROSENTRY RUNTIME ENGINE TELEMETRY DASHBOARD':^65}")
    print("="*65)
    print(f"{'Unit':<6} | {'Cycle':<6} | {'Setting 1':<10} | {'Sensor 2':<10} | {'Sensor 3':<10}")
    print("-"*65)

    try:
        while True:
            data = pipeline_socket.recv(4096).decode('utf-8')
            if not data:
                print("\n[+] Telemetry socket concluded stream. Pipeline tearing down cleanly.")
                break
            
            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue
                
                raw_values = [float(x) for x in line.split() if x.strip()]
                
                if len(raw_values) == len(CMAPSS_COLUMNS):
                    # Formulate structured row frame element
                    df_row = pd.DataFrame([raw_values], columns=CMAPSS_COLUMNS)
                    
                    unit = int(df_row['unit_number'].iloc[0])
                    cycle = int(df_row['time_in_cycles'].iloc[0])
                    s1 = df_row['setting_1'].iloc[0]
                    s2 = df_row['sensor_measurement_2'].iloc[0]
                    s3 = df_row['sensor_measurement_3'].iloc[0]
                    
                    print(f"{unit:<6} | {cycle:<6} | {s1:<10.4f} | {s2:<10.2f} | {s3:<10.2f}")
                    
    except KeyboardInterrupt:
        print("\n[-] Pipeline execution manually halted via user keyboard break.")
    finally:
        pipeline_socket.close()

if __name__ == '__main__':
    launch_pipeline()