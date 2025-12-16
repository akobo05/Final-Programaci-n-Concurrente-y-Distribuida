import socket
import struct
import random
import time

# Configuration
NODES = [('localhost', 5000), ('localhost', 5001)]
IMG_SIZE = 32
INPUT_SIZE = IMG_SIZE * IMG_SIZE
NUM_SAMPLES = 10
ITERATIONS = 1000

def read_exactly(sock, n):
    """
    Helper function to receive exactly n bytes from the socket.
    Raises an exception if connection is closed before reading n bytes.
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Connection closed while reading data")
        data += packet
    return data

def get_active_leader_socket():
    """
    Attempts to connect to available nodes (Failover logic).
    Returns a connected socket or None.
    """
    for host, port in NODES:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0) # Fast timeout for discovery
            s.connect((host, port))
            s.settimeout(None) # Reset timeout for operations
            return s
        except OSError:
            s.close()
    return None

def generate_random_data():
    """Generates random training data simulating a small file."""
    inputs = []
    targets = []
    for _ in range(NUM_SAMPLES):
        vec = [random.random() for _ in range(INPUT_SIZE)]
        inputs.append(vec)
        targets.append(float(random.randint(0, 1)))
    return inputs, targets

def run_benchmark():
    print(f"Starting Benchmark RAFT...")
    print(f"Target Iterations: {ITERATIONS}")
    print(f"Samples per iteration: {NUM_SAMPLES}")
    print(f"Failover Nodes: {NODES}")
    print("-" * 40)

    model_ids = []
    start_time = time.time()

    for i in range(ITERATIONS):
        # Generate data
        inputs, targets = generate_random_data()

        # Prepare payload
        # Protocol: [0x02] [Length] [NumSamples] [InputSize] [Inputs...] [Targets...]
        payload_body = struct.pack('>I', NUM_SAMPLES) + struct.pack('>I', INPUT_SIZE)
        input_bytes = b''.join([struct.pack('>d', x) for vec in inputs for x in vec])
        target_bytes = b''.join([struct.pack('>d', y) for y in targets])

        payload = payload_body + input_bytes + target_bytes
        length = len(payload)

        success = False
        while not success:
            s = None
            try:
                s = get_active_leader_socket()
                if not s:
                    print(f"\rIteration {i+1}: No active leader found. Retrying...", end='')
                    time.sleep(1)
                    continue

                # Set a timeout for the transaction to detect stuck nodes (e.g. unresponsive candidate)
                s.settimeout(5.0)

                with s:
                    # Send request
                    s.sendall(b'\x02')
                    s.sendall(struct.pack('>I', length))
                    s.sendall(payload)

                    # Receive response
                    # Read length (2 bytes)
                    response_len_data = read_exactly(s, 2)

                    msg_len = struct.unpack('>H', response_len_data)[0]
                    msg_bytes = read_exactly(s, msg_len)
                    msg = msg_bytes.decode('utf-8')

                    if "Model ID:" in msg:
                        try:
                            model_id = int(msg.split("Model ID:")[1].strip())
                            model_ids.append(model_id)
                            success = True
                            print(f"\rIteration {i+1}/{ITERATIONS} - Model ID: {model_id}   ", end='')
                        except ValueError:
                             print(f"\rIteration {i+1}: Error parsing ID from '{msg}'", end='')
                    else:
                         print(f"\rIteration {i+1}: Unexpected response: {msg}", end='')
                         # Check if it was an error message
                         if "Error" in msg:
                             # If it's a hard error from server, we might count it as failed or retry?
                             # For benchmark, we retry until success.
                             time.sleep(0.5)

            except socket.timeout:
                print(f"\rIteration {i+1}: Socket timeout (Node stuck?). Retrying...", end='')
            except Exception as e:
                print(f"\rIteration {i+1}: Connection Error ({e}). Retrying...", end='')
                time.sleep(0.5)
            finally:
                if s:
                    s.close()

    total_time = time.time() - start_time
    print("\n\n" + "=" * 40)
    print("Benchmark Completed.")
    print("=" * 40)

    # Metrics
    if ITERATIONS > 0:
        avg_time = total_time / ITERATIONS
        throughput = ITERATIONS / total_time
    else:
        avg_time = 0
        throughput = 0

    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Average Time per Model: {avg_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} models/second")

    # Validation
    print("\nValidating Consensus...")
    model_ids.sort()

    if not model_ids:
        print("No models created.")
        return

    first_id = model_ids[0]
    last_id = model_ids[-1]

    gaps = []
    duplicates = []

    seen = set()
    for idx, mid in enumerate(model_ids):
        if mid in seen:
            duplicates.append(mid)
        seen.add(mid)

        if idx > 0:
            prev = model_ids[idx-1]
            if mid != prev + 1 and mid != prev:
                 # If mid is strictly greater than prev+1, there is a gap
                 if mid > prev + 1:
                     gaps.append((prev, mid))

    # Filter duplicates from gaps check logic
    unique_ids = sorted(list(set(model_ids)))
    real_gaps = []
    for i in range(len(unique_ids) - 1):
        if unique_ids[i+1] != unique_ids[i] + 1:
            real_gaps.append((unique_ids[i], unique_ids[i+1]))

    if not real_gaps and not duplicates:
        print("VALIDATION SUCCESS: IDs are sequential with no gaps or duplicates.")
        print(f"ID Range: {first_id} -> {last_id}")
    else:
        print("VALIDATION FAILED:")
        if real_gaps:
            print(f"Gaps found: {real_gaps}")
        if duplicates:
            print(f"Duplicates found: {duplicates}")

if __name__ == "__main__":
    run_benchmark()
