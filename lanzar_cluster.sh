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

echo "👑 Iniciando Líder..."
# El Líder ya no conecta a los workers, espera conexiones.
java -cp java_leader NodoLider &

sleep 2

echo "🚀 Lanzando $NUM_WORKERS Workers que conectarán al Líder..."

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    W_PORT=$((BASE_PORT_WORKER + i))
    WEB_PORT=$((BASE_PORT_WEB + i))

    # Lanzar cada worker con puertos únicos (para identidad y web)
    # Se conectan automáticamente a localhost:9000
    ./cpp_worker/worker $W_PORT $WEB_PORT &
    echo "  ✅ Worker $i (ID: $W_PORT, Web: $WEB_PORT)"
done

sleep 2

echo "📱 Iniciando Cliente..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python python_client/cliente.py &

wait
