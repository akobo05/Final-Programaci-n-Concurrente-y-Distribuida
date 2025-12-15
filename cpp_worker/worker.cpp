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

#define PORT_WORKER 5001
#define PORT_WEB 8081
#define BUFFER_SIZE 4096

// Protocol Commands
const uint8_t CMD_HEARTBEAT = 0x01;
const uint8_t CMD_DISTRIBUTE_CHUNK = 0x03;
const uint8_t CMD_TRAIN_RESULT = 0x04;

// Global state for monitoring
int processed_chunks = 0;
std::mutex stats_mutex;

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

void handle_client(int client_socket) {
    std::cout << "Leader connected." << std::endl;

    struct timeval tv;
    tv.tv_sec = 10;
    tv.tv_usec = 0;
    setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

    while (true) {
        uint8_t header;
        if (!read_n_bytes(client_socket, &header, 1)) {
            std::cerr << "Leader disconnected or timeout." << std::endl;
            break;
        }

        if (header == CMD_HEARTBEAT) {
            uint32_t dummy; read_n_bytes(client_socket, &dummy, 4); // Read length (0)
            // std::cout << "[RAFT] Heartbeat received." << std::endl;
        } 
        else if (header == CMD_DISTRIBUTE_CHUNK) {
            uint32_t net_len;
            read_n_bytes(client_socket, &net_len, 4);
            uint32_t length = ntohl(net_len);
            std::vector<char> payload(length);
            read_n_bytes(client_socket, payload.data(), length);

            // Parse Protocol: [NumSamples (4)] [InputSize (4)] [Inputs...] [Targets...]
            int offset = 0;
            uint32_t num_samples = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;
            uint32_t input_size = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;

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
            // For 32x32 images, input_size = 1024.
            std::vector<double> w(input_size, 0.5); // Fixed start
            double b = 0.5;
            
            std::vector<double> grad_w(input_size, 0.0);
            double grad_b = 0.0;

            for(int i=0; i<num_samples; i++) {
                double pred = b;
                for(int j=0; j<input_size; j++) {
                    pred += w[j] * inputs[i][j];
                }
                
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
        else if (header == 0x07) { // COMMIT_ENTRY
            uint32_t net_len;
            read_n_bytes(client_socket, &net_len, 4);
            uint32_t length = ntohl(net_len);
            std::vector<char> payload(length);
            read_n_bytes(client_socket, payload.data(), length);

            int offset = 0;
            uint32_t model_id = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;
            
            // Read Weights Vector
            // Payload: [ModelID] [InputSize] [Weights...] [Bias]
            // Wait, previous implementation was just w, b. Now it's vector.
            // Let's assume Leader sends InputSize too.
            // Actually, let's just read until end - 8 bytes.
            // Or better, Leader should send InputSize.
            
            // Let's assume standard protocol for COMMIT: [ModelID] [InputSize] [W...] [b]
            uint32_t input_size = ntohl(*(uint32_t*)(payload.data() + offset)); offset += 4;
            
            std::vector<double> w;
            for(int i=0; i<input_size; i++) {
                uint64_t temp;
                std::memcpy(&temp, payload.data() + offset, 8); offset += 8;
                w.push_back(ntohd(temp));
            }
            
            uint64_t temp_b;
            std::memcpy(&temp_b, payload.data() + offset, 8); offset += 8;
            double b = ntohd(temp_b);

            std::cout << "[RAFT] Commit Entry: Model " << model_id << " (InputSize=" << input_size << ")" << std::endl;
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
    address.sin_port = htons(PORT_WEB);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("web bind failed");
        return NULL;
    }

    if (listen(server_fd, 3) < 0) {
        perror("web listen");
        return NULL;
    }

    std::cout << "Web Monitor listening on port " << PORT_WEB << std::endl;

    while (true) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            perror("web accept");
            continue;
        }

        std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n";
        response += "<html><body><h1>Worker Node (C++)</h1>";
        response += "<p>Status: <b>FOLLOWER</b></p>";
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            response += "<p>Chunks Processed: " + std::to_string(processed_chunks) + "</p>";
        }
        response += "</body></html>";

        send(new_socket, response.c_str(), response.length(), 0);
        close(new_socket);
    }
    return NULL;
}

int main() {
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
    address.sin_port = htons(PORT_WORKER);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    std::cout << "Worker listening on port " << PORT_WORKER << std::endl;

    while (true) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            continue;
        }
        handle_client(new_socket);
    }

    return 0;
}
