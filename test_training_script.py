import socket
import struct
import time
import os
from PIL import Image

def test_training():
    host = 'localhost'
    port = 5000 # Leader

    # Create fake dataset
    num_samples = 10
    input_size = 32*32

    # [0x02] [Length] [NumSamples] [InputSize] [Inputs...] [Targets...]

    payload = struct.pack('>I', num_samples) + struct.pack('>I', input_size)

    for _ in range(num_samples):
        # Fake image data (random or zero)
        for _ in range(input_size):
            payload += struct.pack('>d', 0.5)

    for _ in range(num_samples):
        # Target
        payload += struct.pack('>d', 1.0)

    length = len(payload)

    print("Connecting to leader...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    print("Sending training request...")
    s.sendall(b'\x02')
    s.sendall(struct.pack('>I', length))
    s.sendall(payload)

    print("Waiting for response (this might take a while for 50 epochs)...")

    # Read response
    # Java writes UTF: [Len(2)] [Bytes...]
    # But wait, logic in client is:
    # response_data = s.recv(1024)
    # msg_len = struct.unpack('>H', response_data[:2])[0]

    # Let's read length first
    len_bytes = s.recv(2)
    msg_len = struct.unpack('>H', len_bytes)[0]
    msg_bytes = s.recv(msg_len)
    msg = msg_bytes.decode('utf-8')

    print(f"Server Response: {msg}")
    s.close()

if __name__ == "__main__":
    test_training()
