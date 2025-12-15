# Sistema de Entrenamiento Distribuido de IA (RAFT)

Este proyecto implementa un sistema distribuido tolerante a fallos para el entrenamiento de modelos de IA, utilizando el algoritmo de consenso RAFT (simplificado).

## Arquitectura

*   **Líder (Java):** Coordina el cluster, divide los datos y mantiene el estado RAFT.
*   **Worker (C++):** Procesa fragmentos de datos matemáticamente a alta velocidad.
*   **Cliente (Python):** Interfaz gráfica para cargar datos y controlar el entrenamiento.

## Requisitos

*   Java JDK 8 o superior.
*   G++ (Compilador de C++).
*   Python 3 con Tkinter (`sudo apt-get install python3-tk`).
*   Librería Pillow (`pip install -r python_client/requirements.txt`).

## Instrucciones de Compilación y Ejecución

Es importante seguir el orden de ejecución para asegurar la conexión correcta de los sockets.

### 1. Compilar y Ejecutar el Worker (C++)

Abre una terminal y ejecuta:

```bash
# Compilar
g++ -pthread -o cpp_worker/worker cpp_worker/worker.cpp

# Ejecutar
./cpp_worker/worker
```
*Debe mostrar: `Worker listening on port 5001`*

### 2. Compilar y Ejecutar el Líder (Java)

Abre una **segunda terminal** y ejecuta:

```bash
# Compilar
javac java_leader/NodoLider.java

# Ejecutar
java -cp java_leader NodoLider
```
*Debe mostrar: `Connected to Worker...` y comenzar a enviar Heartbeats.*

### 3. Ejecutar el Cliente (Python)

Abre una **tercera terminal** y ejecuta:

```bash
python3 python_client/cliente.py
```

## Uso del Sistema

## Uso del Sistema
**Nota:** El cliente requiere la librería `Pillow` para cargar imágenes: `pip install Pillow`.

1.  **Entrenamiento:**
    *   Prepara una carpeta con tu dataset. Debe contener dos subcarpetas: `0` y `1` (para clasificación binaria).
    *   En la sección "Entrenamiento", haz clic en **"Seleccionar Carpeta Dataset"**.
    *   El sistema cargará las imágenes, las redimensionará a 32x32 y las convertirá a vectores.
    *   Haz clic en **"Iniciar Entrenamiento"**.
    *   Al finalizar, aparecerá una ventana emergente para que le asignes un **Nombre** al modelo (ej: "Modelo V1").

2.  **Testeo / Inferencia:**
    *   En la sección "Testeo", selecciona el modelo que deseas probar de la **lista desplegable**.
    *   Si no aparece, haz clic en el botón de **Refrescar (↻)**.
    *   Haz clic en **"Cargar Imagen..."** y selecciona una imagen (JPG/PNG).
    *   El sistema enviará la imagen al Líder y te devolverá la predicción.

## Monitoreo Web

Puedes ver el estado de los nodos desde tu navegador:

*   **Líder:** [http://localhost:8080](http://localhost:8080)
*   **Worker:** [http://localhost:8081](http://localhost:8081)
