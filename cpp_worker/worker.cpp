/*
 * Compile with: g++ -pthread -o worker worker.cpp
 */

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <algorithm>
#include <map>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <chrono>
#include <endian.h>

#define BUFFER_SIZE 4096
#define LEADER_PORT 9000
#define LEADER_IP "127.0.0.1"

// Defaults
int port_worker = 5001; // Used for Identity/File naming
int port_web = 8081;

// Protocol Commands
const uint8_t CMD_HEARTBEAT = 0x01;
const uint8_t CMD_DISTRIBUTE_CHUNK = 0x03;
const uint8_t CMD_TRAIN_RESULT = 0x04;
const uint8_t CMD_PREDICT = 0x06;
const uint8_t CMD_COMMIT_ENTRY = 0x07;

// Data Structures
struct Model {
    uint32_t input_size;
    std::vector<double> w;
    double b;
};

// RAFT State (Kept for compatibility, though acting as Client now)
enum RaftState { FOLLOWER, CANDIDATE, LEADER };

// Global state
std::atomic<RaftState> state(FOLLOWER);
std::atomic<int> current_term(0);
std::atomic<long long> last_heartbeat_time(0);

int processed_chunks = 0;
std::mutex stats_mutex;

std::map<int, Model> model_store;
std::mutex model_mutex;
std::string state_filename = "worker_data.dat"; // Updated in main

// Helper: Get current time in milliseconds
long long current_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

// Election Timer Thread
void election_timer() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        long long now = current_time_ms();

        // Check for timeout (5 seconds)
        if (now - last_heartbeat_time > 5000) {
            if (state == FOLLOWER) {
                state = CANDIDATE;
                current_term++;
                // std::cout << "Leader Dead! Promoting to CANDIDATE..." << std::endl;
                last_heartbeat_time = current_time_ms();
            }
        }
    }
}

// Helper to read exactly n bytes
bool read_n_bytes(int socket, void* buffer, size_t n) {
    size_t total_read = 0;
    char* ptr = (char*)buffer;
    while (total_read < n) {
        ssize_t bytes_read = recv(socket, ptr + total_read, n - total_read, 0);
        if (bytes_read <= 0) return false;
        total_read += bytes_read;
    }
    return true;
}

// Helper to handle Endianness for Doubles
double ntohd(uint64_t val) {
    uint64_t host_val = be64toh(val);
    double res;
    std::memcpy(&res, &host_val, sizeof(double));
    return res;
}

uint64_t htond(double val) {
    uint64_t bits;
    std::memcpy(&bits, &val, sizeof(double));
    return htobe64(bits);
}

// Math Implementation
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Save state to disk
void save_state() {
    std::lock_guard<std::mutex> lock(model_mutex);
    std::ofstream outfile(state_filename, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "[Error] Could not save state to " << state_filename << std::endl;
        return;
    }

    uint32_t count = htonl(model_store.size());
    outfile.write((char*)&count, 4);

    for (auto const& [id, model] : model_store) {
        uint32_t net_id = htonl(id);
        uint32_t net_size = htonl(model.input_size);

        outfile.write((char*)&net_id, 4);
        outfile.write((char*)&net_size, 4);

        for (double val : model.w) {
            uint64_t net_val = htond(val);
            outfile.write((char*)&net_val, 8);
        }

        uint64_t net_b = htond(model.b);
        outfile.write((char*)&net_b, 8);
    }
    outfile.close();
}

// Load state from disk
void load_state() {
    std::ifstream infile(state_filename, std::ios::binary);
    if (!infile.is_open()) return;

    uint32_t net_count;
    if (!infile.read((char*)&net_count, 4)) return;
    uint32_t count = ntohl(net_count);

    for (uint32_t i = 0; i < count; i++) {
        uint32_t net_id, net_size;
        infile.read((char*)&net_id, 4);
        infile.read((char*)&net_size, 4);

        int id = ntohl(net_id);
        int size = ntohl(net_size);

        std::vector<double> w;
        w.reserve(size);
        for(int j=0; j<size; j++) {
            uint64_t net_val;
            infile.read((char*)&net_val, 8);
            w.push_back(ntohd(net_val));
        }

        uint64_t net_b;
        infile.read((char*)&net_b, 8);
        double b = ntohd(net_b);

        model_store[id] = { (uint32_t)size, w, b };
    }
    std::cout << "[State] Loaded " << count << " models from disk (" << state_filename << ")." << std::endl;
}

void send_utf_string(int socket, std::string msg) {
    uint16_t len = msg.length();
    uint16_t net_len = htons(len);
    send(socket, &net_len, 2, 0);
    send(socket, msg.c_str(), len, 0);
}

// Command Loop (Renamed from handle_client)
void process_commands(int socket_fd) {
    std::cout << "Connected to Leader. Waiting for commands..." << std::endl;

    // Send Handshake (Optional, but good for ID).
    // Sending Port so Leader knows who we are (visuals).
    uint32_t my_port = htonl(port_worker);
    send(socket_fd, &my_port, 4, 0);

    while (true) {
        uint8_t header;
        if (!read_n_bytes(socket_fd, &header, 1)) {
            break; // Connection lost
        }

        if (header == CMD_HEARTBEAT) {
            uint32_t dummy;
            if (read_n_bytes(socket_fd, &dummy, 4)) {
                last_heartbeat_time = current_time_ms();
                if (state == CANDIDATE || state == LEADER) {
                    state = FOLLOWER;
                }
            }
        } 
        else if (header == CMD_DISTRIBUTE_CHUNK) {
            uint32_t net_len;
            if (!read_n_bytes(socket_fd, &net_len, 4)) break;
            uint32_t length = ntohl(net_len);
            std::vector<char> payload(length);
            if (!read_n_bytes(socket_fd, payload.data(), length)) break;

            // Parse Protocol: [NumSamples (4)] [InputSize (4)] [W...] [b] [Inputs...] [Targets...]
            if (length < 8) {
                std::cerr << "Invalid DISTRIBUTE_CHUNK payload length." << std::endl;
                continue;
            }
            int offset = 0;
            uint32_t num_samples = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;
            uint32_t input_size = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;

            // Calculate expected size with Weights and Bias
            size_t expected_size = 8 +
                                   (size_t)input_size * 8 + // W
                                   8 +                      // b
                                   (size_t)num_samples * input_size * 8 + // Inputs
                                   (size_t)num_samples * 8;               // Targets

            if (length < expected_size) {
                 std::cerr << "Invalid DISTRIBUTE_CHUNK payload size." << std::endl;
                 continue;
            }

            // Read Weights
            std::vector<double> w;
            w.reserve(input_size);
            for(int j=0; j<input_size; j++) {
                uint64_t temp;
                std::memcpy(&temp, payload.data() + offset, 8); offset += 8;
                w.push_back(ntohd(temp));
            }

            // Read Bias
            uint64_t temp_b;
            std::memcpy(&temp_b, payload.data() + offset, 8); offset += 8;
            double b = ntohd(temp_b);

            std::vector<std::vector<double>> inputs(num_samples, std::vector<double>(input_size));
            std::vector<double> targets(num_samples);

            for(int i=0; i<num_samples; i++) {
                for(int j=0; j<input_size; j++) {
                    uint64_t temp;
                    std::memcpy(&temp, payload.data() + offset, 8); offset += 8;
                    inputs[i][j] = ntohd(temp);
                }
            }
            for(int i=0; i<num_samples; i++) {
                uint64_t temp;
                std::memcpy(&temp, payload.data() + offset, 8); offset += 8;
                targets[i] = ntohd(temp);
            }

            // std::cout << "[WORKER] Training on " << num_samples << " images (size " << input_size << ")..." << std::endl;

            // Linear Regression Gradient Descent (Vectorized)
            // Model: y = W.x + b
            
            std::vector<double> grad_w(input_size, 0.0);
            double grad_b = 0.0;

            for(int i=0; i<num_samples; i++) {
                double z = b;
                for(int j=0; j<input_size; j++) {
                    z += w[j] * inputs[i][j];
                }
                
                // 1. Apply Sigmoid
                double pred = sigmoid(z);
                
                // 2. Calculate Error
                double error = targets[i] - pred;

                // 3. Accumulate Gradients
                // dL/dw = (pred - y) * x = -error * x
                // We keep the -2 factor as requested/legacy style or just standard gradient?
                // Standard: grad += (pred - y) * x
                // Previous code was: grad += -2 * x * error.
                // Since error = y - pred, -error = pred - y.
                // So -2 * x * (y - p) = 2 * x * (p - y).
                // It's just a scaled gradient. I will keep it to minimize friction with Leader logic.
                for(int j=0; j<input_size; j++) {
                    grad_w[j] += -2 * inputs[i][j] * error;
                }
                grad_b += -2 * error;
            }

            // Average gradients
            for(int j=0; j<input_size; j++) grad_w[j] /= num_samples;
            grad_b /= num_samples;

            // Send Gradients back
            uint8_t resp_header = CMD_TRAIN_RESULT;
            uint32_t resp_len = htonl((input_size * 8) + 8); 
            
            send(socket_fd, &resp_header, 1, 0);
            send(socket_fd, &resp_len, 4, 0);
            
            for(int j=0; j<input_size; j++) {
                uint64_t net_gw = htond(grad_w[j]);
                send(socket_fd, &net_gw, 8, 0);
            }
            uint64_t net_gb = htond(grad_b);
            send(socket_fd, &net_gb, 8, 0);

            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                processed_chunks++;
            }
        }
        else if (header == CMD_COMMIT_ENTRY) {
            uint32_t net_len;
            if (!read_n_bytes(socket_fd, &net_len, 4)) break;
            uint32_t length = ntohl(net_len);

            std::vector<char> payload(length);
            if (!read_n_bytes(socket_fd, payload.data(), length)) break;

            int offset = 0;
            uint32_t model_id = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;
            uint32_t input_size = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;
            
            std::vector<double> w;
            w.reserve(input_size);
            for(int i=0; i<input_size; i++) {
                uint64_t temp;
                std::memcpy(&temp, payload.data() + offset, 8); offset += 8;
                w.push_back(ntohd(temp));
            }
            
            uint64_t temp_b;
            std::memcpy(&temp_b, payload.data() + offset, 8); offset += 8;
            double b = ntohd(temp_b);

            {
                std::lock_guard<std::mutex> lock(model_mutex);
                model_store[model_id] = { input_size, w, b };
            }
            save_state();
            std::cout << "[RAFT] Committed Model " << model_id << std::endl;
        }
        else if (header == CMD_PREDICT) {
            // Worker usually doesn't receive PREDICT from Leader in this architecture?
            // Unless Leader acts as proxy.
            // We implement it just in case.
            uint32_t net_len;
            if (!read_n_bytes(socket_fd, &net_len, 4)) break;
            uint32_t length = ntohl(net_len);
            std::vector<char> payload(length);
            if (!read_n_bytes(socket_fd, payload.data(), length)) break;

            int offset = 0;
            uint32_t model_id = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;
            int input_size = (length - 4) / 8;

            std::vector<double> input_val;
            for(int i=0; i<input_size; i++) {
                uint64_t temp;
                std::memcpy(&temp, payload.data() + offset, 8); offset += 8;
                input_val.push_back(ntohd(temp));
            }

            std::string response;
            {
                std::lock_guard<std::mutex> lock(model_mutex);
                if (model_store.count(model_id)) {
                    const Model& m = model_store[model_id];
                    double z = m.b;
                    for(int i=0; i<input_size; i++) z += m.w[i] * input_val[i];
                    double prob = sigmoid(z);
                    std::ostringstream oss;
                    oss << (prob > 0.5 ? "Dígito 1" : "Dígito 0")
                        << " (Prob: " << std::fixed << std::setprecision(2) << prob << ")";
                    response = oss.str();
                } else {
                    response = "Error: Model not found";
                }
            }
            send_utf_string(socket_fd, response);
        }
    }
}

void* web_server(void* arg) {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) return NULL;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) return NULL;

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_web);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) return NULL;
    if (listen(server_fd, 3) < 0) return NULL;

    std::cout << "Web Monitor listening on port " << port_web << std::endl;

    while (true) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) continue;

        std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n";
        response += "<html><body><h1>Worker Node (C++) - " + std::to_string(port_worker) + "</h1>";
        response += "<p>Mode: <b>Connected Worker</b></p>";
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            response += "<p>Chunks Processed: " + std::to_string(processed_chunks) + "</p>";
        }
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            response += "<p>Models Stored: " + std::to_string(model_store.size()) + "</p>";
        }
        response += "</body></html>";
        send(new_socket, response.c_str(), response.length(), 0);
        close(new_socket);
    }
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc >= 2) port_worker = std::stoi(argv[1]);
    if (argc >= 3) port_web = std::stoi(argv[2]);

    // Dynamic State Filename
    state_filename = "worker_" + std::to_string(port_worker) + ".dat";

    // Load state
    load_state();

    // Start Web Server Thread
    pthread_t web_thread;
    pthread_create(&web_thread, NULL, web_server, NULL);

    // Infinite Life Loop with Reconnection
    while(true) {
        int sock = 0;
        struct sockaddr_in serv_addr;

        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            perror("Socket creation error");
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(LEADER_PORT);

        if (inet_pton(AF_INET, LEADER_IP, &serv_addr.sin_addr) <= 0) {
            perror("Invalid address/ Address not supported");
            close(sock);
            return -1;
        }

        // Try to connect
        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
            // Connection failed
            // std::cout << "Connection Failed. Retrying in 5s..." << std::endl;
            close(sock);
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        // Connected!
        // Handle connection until it breaks
        process_commands(sock);

        // Connection broken
        close(sock);
        std::cout << "Conexión perdida. Reintentando en 5s..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    return 0;
}
