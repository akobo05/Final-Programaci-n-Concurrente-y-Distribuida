import os
import gzip
import struct
import urllib.request
from PIL import Image

def preparar_mnist_ligero():
    # URL de un mirror confiable de MNIST
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "images": "train-images-idx3-ubyte.gz",
        "labels": "train-labels-idx1-ubyte.gz"
    }

    # 1. Descargar archivos binarios si no existen
    for filename in files.values():
        if not os.path.exists(filename):
            print(f"Descargando {filename} (esto es rápido, ~10MB)...")
            urllib.request.urlretrieve(base_url + filename, filename)

    # 2. Crear carpetas 0 y 1
    base_dir = "dataset_mnist"
    for cls in ['0', '1']:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

    print("Extrayendo ceros y unos...")

    # 3. Leer Etiquetas
    with gzip.open(files["labels"], 'rb') as f:
        # Los primeros 8 bytes son el encabezado (magic number y cantidad)
        magic, num = struct.unpack(">II", f.read(8))
        labels = list(f.read())

    # 4. Leer Imágenes y guardar solo 0s y 1s
    with gzip.open(files["images"], 'rb') as f:
        # Los primeros 16 bytes son el encabezado
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        
        count = 0
        for i in range(num):
            # Cada imagen mide 28x28 = 784 bytes
            image_data = f.read(rows * cols)
            label = str(labels[i])

            if label in ['0', '1']:
                # Crear imagen desde los bytes crudos usando Pillow
                img = Image.frombytes('L', (cols, rows), image_data)
                img.save(os.path.join(base_dir, label, f"mnist_{i}.png"))
                count += 1
                
                # Opcional: limitar a 1000 imágenes para que sea instantáneo
                if count >= 1000: break

    print(f"✅ Finalizado. {count} imágenes guardadas en '{base_dir}'.")
    print("Ya puedes borrar los archivos .gz si deseas.")

if __name__ == "__main__":
    preparar_mnist_ligero()