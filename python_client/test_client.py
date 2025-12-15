import socket
import struct
import random
import time

LEADER_HOST = 'localhost'
LEADER_PORT = 5000

def run_test():
    print("Generating data (y = 2x + 1)...")
    inputs = [random.uniform(0, 10) for _ in range(1000)]
    targets = [(2 * x + 1) for x in inputs]
    
    print("Connecting to Leader for TRAINING...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((LEADER_HOST, LEADER_PORT))
            
            # Protocol: [0x02] [Length] [NumSamples] [InputSize] [Inputs...] [Targets...]
            header = b'\x02'
            num_samples = len(inputs)
            input_size = 1
            
            payload = struct.pack('>I', num_samples) + struct.pack('>I', input_size)
            payload += b''.join([struct.pack('>d', x) for x in inputs])
            payload += b''.join([struct.pack('>d', y) for y in targets])
            
            length = len(payload)
            
            s.sendall(header)
            s.sendall(struct.pack('>I', length))
            s.sendall(payload)

            print("Training Data sent. Waiting for response...")
            
            response_data = s.recv(1024)
            if len(response_data) > 2:
                msg_len = struct.unpack('>H', response_data[:2])[0]
                msg = response_data[2:2+msg_len].decode('utf-8')
            else:
                msg = response_data.decode('utf-8', errors='ignore')
            print(f"Leader Response: {msg}")

    except Exception as e:
        print(f"Error Training: {e}")
        return

    print("\nConnecting to Leader for INFERENCE (Model 1)...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((LEADER_HOST, LEADER_PORT))
            
            # Test with x = 5.0. Expected y = 11.0
            model_id = 1
            val_x = 5.0
            
            header = b'\x06'
            payload = struct.pack('>I', model_id) + struct.pack('>d', val_x)
            length = len(payload)

            s.sendall(header)
            s.sendall(struct.pack('>I', length))
            s.sendall(payload)
            
            response_data = s.recv(1024)
            if len(response_data) > 2:
                msg_len = struct.unpack('>H', response_data[:2])[0]
                msg = response_data[2:2+msg_len].decode('utf-8')
            else:
                msg = response_data.decode('utf-8', errors='ignore')
            print(f"Inference Result (x={val_x}): {msg}")

    except Exception as e:
        print(f"Error Inference: {e}")

if __name__ == "__main__":
    time.sleep(2) 
    run_test()
