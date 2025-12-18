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

#define BUFFER_SIZE 4096

// Defaults
int port_worker = 5001;
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

// RAFT State
enum RaftState { FOLLOWER, CANDIDATE, LEADER };

// Global state
std::atomic<RaftState> state(FOLLOWER);
std::atomic<int> current_term(0);
std::atomic<long long> last_heartbeat_time(0);

int processed_chunks = 0;
std::mutex stats_mutex;

std::map<int, Model> model_store;
std::mutex model_mutex;
const std::string STATE_FILE = "worker_data.dat";

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
                std::cout << "Leader Dead! Promoting to CANDIDATE..." << std::endl;
                // Reset timer to avoid spamming the console immediately again,
                // effectively acting as "starting a new election period"
                last_heartbeat_time = current_time_ms();
            } else if (state == CANDIDATE) {
                // If we are already candidate and timeout again, typically we restart election (bump term)
                // The prompt simplifies to "Change state to CANDIDATE", but since we are already there,
                // we might just bump term to simulate a retry.
                // However, to keep it simple and non-spammy:
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

// Helper to handle Endianness for Doubles (Network Big Endian -> Host)
double ntohd(uint64_t val) {
    uint64_t host_val = be64toh(val); // Convert Big Endian to Host
    double res;
    std::memcpy(&res, &host_val, sizeof(double));
    return res;
}

// Helper to handle Endianness for Doubles (Host -> Network Big Endian)
uint64_t htond(double val) {
    uint64_t bits;
    std::memcpy(&bits, &val, sizeof(double));
    return htobe64(bits); // Convert Host to Big Endian
}

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Save state to disk
void save_state() {
    std::lock_guard<std::mutex> lock(model_mutex);
    std::ofstream outfile(STATE_FILE, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "[Error] Could not save state." << std::endl;
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
    // std::cout << "[State] Saved " << model_store.size() << " models." << std::endl;
}

// Load state from disk
void load_state() {
    std::ifstream infile(STATE_FILE, std::ios::binary);
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
    std::cout << "[State] Loaded " << count << " models from disk." << std::endl;
}

void send_utf_string(int socket, std::string msg) {
    uint16_t len = msg.length();
    uint16_t net_len = htons(len);
    send(socket, &net_len, 2, 0);
    send(socket, msg.c_str(), len, 0);
}

void handle_client(int client_socket) {
    std::cout << "Client connected (Socket: " << client_socket << ")" << std::endl;

    struct timeval tv;
    tv.tv_sec = 10;
    tv.tv_usec = 0;
    setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

    while (true) {
        uint8_t header;
        if (!read_n_bytes(client_socket, &header, 1)) {
            // std::cerr << "Client disconnected or timeout." << std::endl;
            break;
        }

        if (header == CMD_HEARTBEAT) {
            uint32_t dummy;
            if (read_n_bytes(client_socket, &dummy, 4)) {
                // Heartbeat received logic
                last_heartbeat_time = current_time_ms();

                if (state == CANDIDATE || state == LEADER) {
                    state = FOLLOWER;
                    std::cout << "Leader detected. Demoting to FOLLOWER." << std::endl;
                }
                // std::cout << "[RAFT] Heartbeat received." << std::endl;
            }
        } 
        else if (header == CMD_DISTRIBUTE_CHUNK) {
            uint32_t net_len;
            if (!read_n_bytes(client_socket, &net_len, 4)) break;
            uint32_t length = ntohl(net_len);
            std::vector<char> payload(length);
            if (!read_n_bytes(client_socket, payload.data(), length)) break;

            // Parse Protocol: [NumSamples (4)] [InputSize (4)] [Inputs...] [Targets...]
            if (length < 8) {
                std::cerr << "Invalid DISTRIBUTE_CHUNK payload length." << std::endl;
                continue;
            }
            int offset = 0;
            uint32_t num_samples = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;
            uint32_t input_size = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;

            size_t expected_size = 8 + (size_t)num_samples * input_size * 8 + (size_t)num_samples * 8;
            if (length < expected_size) {
                 std::cerr << "Invalid DISTRIBUTE_CHUNK payload size." << std::endl;
                 continue;
            }

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

            std::cout << "[WORKER] Training on " << num_samples << " images (size " << input_size << ")..." << std::endl;

            // Linear Regression Gradient Descent (Vectorized)
            // Model: y = W.x + b
            std::vector<double> w(input_size, 0.5); // Fixed start
            double b = 0.5;
            
            std::vector<double> grad_w(input_size, 0.0);
            double grad_b = 0.0;

            for(int i=0; i<num_samples; i++) {
                double z = b;
                for(int j=0; j<input_size; j++) {
                    z += w[j] * inputs[i][j];
                }
                
                double pred = sigmoid(z);
                double error = targets[i] - pred;
                
                for(int j=0; j<input_size; j++) {
                    grad_w[j] += -2 * inputs[i][j] * error;
                }
                grad_b += -2 * error;
            }

            // Average gradients
            for(int j=0; j<input_size; j++) grad_w[j] /= num_samples;
            grad_b /= num_samples;

            // Send Gradients back
            // Payload: [grad_w (vector)] [grad_b]
            uint8_t resp_header = CMD_TRAIN_RESULT;
            uint32_t resp_len = htonl((input_size * 8) + 8); 
            
            send(client_socket, &resp_header, 1, 0);
            send(client_socket, &resp_len, 4, 0);
            
            for(int j=0; j<input_size; j++) {
                uint64_t net_gw = htond(grad_w[j]);
                send(client_socket, &net_gw, 8, 0);
            }
            uint64_t net_gb = htond(grad_b);
            send(client_socket, &net_gb, 8, 0);

            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                processed_chunks++;
            }
        }
        else if (header == CMD_COMMIT_ENTRY) { // 0x07
            uint32_t net_len;
            if (!read_n_bytes(client_socket, &net_len, 4)) break;
            uint32_t length = ntohl(net_len);

            // Limit max payload to avoid memory exhaustion
            if (length > 100 * 1024 * 1024) { // 100MB limit
                 std::cerr << "Payload too large." << std::endl;
                 break;
            }

            std::vector<char> payload(length);
            if (!read_n_bytes(client_socket, payload.data(), length)) break;

            if (length < 8) {
                 std::cerr << "Invalid COMMIT_ENTRY payload." << std::endl;
                 continue;
            }

            int offset = 0;
            uint32_t model_id = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;
            uint32_t input_size = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;
            
            // Check if length matches input_size
            // Payload: ID(4) + Size(4) + W(Size*8) + b(8)
            size_t expected_size = 4 + 4 + (size_t)input_size * 8 + 8;
            if (length != expected_size) {
                 std::cerr << "Invalid COMMIT_ENTRY size (Expected " << expected_size << ", got " << length << ")." << std::endl;
                 continue;
            }

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

            std::cout << "[RAFT] Commit Entry: Model " << model_id << " (InputSize=" << input_size << ") Saved." << std::endl;
        }
        else if (header == CMD_PREDICT) { // 0x06
            uint32_t net_len;
            if (!read_n_bytes(client_socket, &net_len, 4)) break;
            uint32_t length = ntohl(net_len);

            if (length > 10 * 1024 * 1024) { // 10MB limit
                 std::cerr << "Payload too large." << std::endl;
                 break;
            }

            std::vector<char> payload(length);
            if (!read_n_bytes(client_socket, payload.data(), length)) break;

            if (length < 4) {
                 std::cerr << "Invalid PREDICT payload." << std::endl;
                 continue;
            }

            int offset = 0;
            uint32_t model_id = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;

            // Calculate remaining bytes for input vector
            int remaining = length - 4;
            if (remaining % 8 != 0) {
                 std::cerr << "Invalid PREDICT input vector size." << std::endl;
                 send_utf_string(client_socket, "Error: Invalid input data.");
                 continue;
            }
            int input_size = remaining / 8;

            std::vector<double> input_val;
            input_val.reserve(input_size);
            for(int i=0; i<input_size; i++) {
                uint64_t temp;
                std::memcpy(&temp, payload.data() + offset, 8); offset += 8;
                input_val.push_back(ntohd(temp));
            }

            std::string response;
            {
                std::lock_guard<std::mutex> lock(model_mutex);
                if (model_store.find(model_id) != model_store.end()) {
                    const Model& m = model_store[model_id];
                    if (m.input_size != (uint32_t)input_size) {
                        response = "Error: Input size mismatch.";
                    } else {
                        double prediction = m.b;
                        for(int i=0; i<input_size; i++) {
                            prediction += m.w[i] * input_val[i];
                        }
                        std::ostringstream oss;
                        oss << "Prediction: " << prediction;
                        response = oss.str();
                    }
                } else {
                    response = "Error: Model " + std::to_string(model_id) + " not found.";
                }
            }
            send_utf_string(client_socket, response);
            // std::cout << "[WORKER] Predict Request (M" << model_id << "): " << response << std::endl;
        }
    }
    close(client_socket);
}

void* web_server(void* arg) {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("web socket failed");
        return NULL;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        return NULL;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_web);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("web bind failed");
        return NULL;
    }

    if (listen(server_fd, 3) < 0) {
        perror("web listen");
        return NULL;
    }

    std::cout << "Web Monitor listening on port " << port_web << std::endl;

    while (true) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            perror("web accept");
            continue;
        }

        std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n";
        response += "<html><body><h1>Worker Node (C++)</h1>";

        // Dynamic State Display
        std::string state_str = "UNKNOWN";
        std::string color = "black";
        RaftState current_s = state.load();

        if (current_s == FOLLOWER) {
            state_str = "FOLLOWER";
            color = "red";
        } else if (current_s == CANDIDATE) {
            state_str = "CANDIDATE";
            color = "orange";
        } else if (current_s == LEADER) {
            state_str = "LEADER";
            color = "green";
        }

        response += "<p>Status: <b style='color:" + color + ";'>" + state_str + "</b></p>";
        response += "<p>Current Term: " + std::to_string(current_term) + "</p>";

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
    if (argc >= 3) {
        port_worker = std::stoi(argv[1]);
        port_web = std::stoi(argv[2]);
    }

    // Initialize Timer
    last_heartbeat_time = current_time_ms();

    load_state();

    // Start Election Timer Thread
    std::thread timer_thread(election_timer);
    timer_thread.detach();

    // Start Web Server Thread
    pthread_t web_thread;
    if (pthread_create(&web_thread, NULL, web_server, NULL) != 0) {
        std::cerr << "Failed to create web thread" << std::endl;
        return 1;
    }

    // Main Socket Server
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_worker);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    std::cout << "Worker initiated on port: " << port_worker << " (Web: " << port_web << ")" << std::endl;

    while (true) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            continue;
        }

        // Multi-threading to handle multiple clients (Leader + Predict Clients)
        std::thread(handle_client, new_socket).detach();
    }

    return 0;
}
