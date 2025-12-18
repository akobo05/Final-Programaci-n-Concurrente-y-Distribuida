#!/bin/bash

# Configuración por defecto
IP_LIDER=${1:-"localhost"}
NUM_WORKERS=${2:-3}  # Lanza 3 workers si no se especifica
BASE_PORT_WORKER=5001
BASE_PORT_WEB=8081

cleanup() {
    echo -e "\n🛑 Deteniendo cluster..."
    pkill -f "cpp_worker/worker"
    pkill -f "NodoLider"
    exit
}
trap cleanup SIGINT

echo "🔨 Compilando..."
g++ -pthread -o cpp_worker/worker cpp_worker/worker.cpp
javac java_leader/NodoLider.java

echo "🚀 Lanzando $NUM_WORKERS Workers en $IP_LIDER..."

WORKER_PORTS=""

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    W_PORT=$((BASE_PORT_WORKER + i))
    WEB_PORT=$((BASE_PORT_WEB + i))

    # Lanzar cada worker con puertos únicos
    ./cpp_worker/worker $W_PORT $WEB_PORT &
    echo "  ✅ Worker $i en puertos $W_PORT y $WEB_PORT"

    WORKER_PORTS="$WORKER_PORTS $W_PORT"
done

sleep 2

echo "👑 Iniciando Líder..."
# El Líder recibe la lista de puertos de los workers
echo "   Connecting Leader to ports: $WORKER_PORTS"
java -cp java_leader NodoLider $WORKER_PORTS &

sleep 2

echo "📱 Iniciando Cliente..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if python client exists and can be run (requires GUI usually, but maybe headless)
# If this is a headless environment, client might fail to open window.
# But I will include it as per instructions.
python python_client/cliente.py &

wait
