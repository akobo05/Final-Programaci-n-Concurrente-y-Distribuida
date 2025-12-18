import socket
import struct
import time
import sys

# Protocol Config
PORT = 5000
HOST = 'localhost'

def connect():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        return s
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

def test_list_workers():
    print("\n--- Testing LIST_WORKERS (0x10) ---")
    s = connect()
    if not s: return
    try:
        s.sendall(b'\x10')
        count = struct.unpack('>I', s.recv(4))[0]
        print(f"Worker count: {count}")
        for _ in range(count):
            len_val = struct.unpack('>H', s.recv(2))[0]
            host = s.recv(len_val).decode('utf-8')
            port = struct.unpack('>I', s.recv(4))[0]
            print(f" - Worker: {host}:{port}")
        return count
    finally:
        s.close()

def test_train():
    print("\n--- Testing START_TRAIN (0x02) ---")
    s = connect()
    if not s: return
    try:
        # Send 2 samples, size 2
        # Inputs: [[0,0], [1,1]], Targets: [0, 1]
        num_samples = 2
        input_size = 2
        inputs = [0.0, 0.0, 1.0, 1.0]
        targets = [0.0, 1.0]

        payload = struct.pack('>I', num_samples) + struct.pack('>I', input_size)
        payload += b''.join([struct.pack('>d', x) for x in inputs])
        payload += b''.join([struct.pack('>d', x) for x in targets])

        length = len(payload)
        s.sendall(b'\x02')
        s.sendall(struct.pack('>I', length))
        s.sendall(payload)

        # Read response
        resp_len = struct.unpack('>H', s.recv(2))[0]
        msg = s.recv(resp_len).decode('utf-8')
        print(f"Response: {msg}")

        if "Model ID:" in msg:
            return int(msg.split("Model ID:")[1].strip())
        return None
    finally:
        s.close()

def test_predict(model_id):
    print(f"\n--- Testing PREDICT (0x06) for Model {model_id} ---")
    s = connect()
    if not s: return
    try:
        # Input [1, 1] -> Should be Class 1
        input_val = [1.0, 1.0]
        payload = struct.pack('>I', model_id)
        payload += b''.join([struct.pack('>d', x) for x in input_val])
        length = len(payload)

        s.sendall(b'\x06')
        s.sendall(struct.pack('>I', length))
        s.sendall(payload)

        resp_len = struct.unpack('>H', s.recv(2))[0]
        msg = s.recv(resp_len).decode('utf-8')
        print(f"Prediction: {msg}")
    finally:
        s.close()

def test_kill(port_to_kill):
    print(f"\n--- Testing KILL_WORKER (0x11) on port {port_to_kill} ---")
    s = connect()
    if not s: return
    try:
        s.sendall(b'\x11')
        s.sendall(struct.pack('>I', port_to_kill))

        resp_len = struct.unpack('>H', s.recv(2))[0]
        msg = s.recv(resp_len).decode('utf-8')
        print(f"Kill Response: {msg}")
    finally:
        s.close()

if __name__ == "__main__":
    time.sleep(2) # Wait for cluster to stabilize

    count = test_list_workers()

    model_id = test_train()

    if model_id:
        test_predict(model_id)

    # Kill the last worker
    if count and count > 0:
        # Assume ports start at 5001
        target_port = 5001 + count - 1
        test_kill(target_port)

        time.sleep(1)
        test_list_workers()
