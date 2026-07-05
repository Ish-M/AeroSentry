import socket
import time
import os

HOST = '127.0.0.1'
PORT = 9999
# Tailored exactly to your directory: data/CMAPPS/train_FD001.txt
DATA_PATH = os.path.join('data', 'CMAPPS', 'train_FD001.txt')

def stream_telemetry():
    if not os.path.exists(DATA_PATH):
        print(f"[-] Error: C-MAPSS data file not found at: {os.path.abspath(DATA_PATH)}")
        print("[-] Please double check your data folder casing matches 'data/CMAPPS/train_FD001.txt'")
        return

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f"[+] AeroSentry Mock Streamer booted live on {HOST}:{PORT}")
    print("[+] Waiting for Data Pipeline to establish handshake...")
    
    conn, addr = server_socket.accept()
    print(f"[+] Data Pipeline linked securely from connection: {addr}")
    
    try:
        with open(DATA_PATH, 'r') as f:
            for line in f:
                conn.sendall(line.encode('utf-8'))
                time.sleep(0.02)  # Simulates active 50Hz sensor stream
        print("[+] Telemetry stream completed successfully for dataset subset FD001.")
    except ConnectionResetError:
        print("[-] Pipeline disconnected unexpectedly mid-stream.")
    finally:
        conn.close()
        server_socket.close()

if __name__ == '__main__':
    stream_telemetry()