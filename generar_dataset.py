import os
import random
from PIL import Image
import shutil

def generar_dataset():
    # 1. Crear estructura de carpetas
    base_dir = os.path.join(os.getcwd(), 'dataset_prueba')

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir) # Limpiar si ya existe

    os.makedirs(base_dir)

    classes = ['0', '1']
    for cls in classes:
        os.makedirs(os.path.join(base_dir, cls))

    # 2. Generar Imágenes
    num_images_per_class = 50
    img_size = (32, 32)

    for cls in classes:
        for i in range(num_images_per_class):
            # Crear imagen aleatoria (Ruido)
            # Modo 'L' para escala de grises (0-255)
            # Generamos datos aleatorios para cada pixel
            pixels = [random.randint(0, 255) for _ in range(img_size[0] * img_size[1])]
            img = Image.new('L', img_size)
            img.putdata(pixels)

            filename = f"img_{cls}_{i}.png"
            filepath = os.path.join(base_dir, cls, filename)
            img.save(filepath)

    # 3. Confirmación
    print(f"Dataset generado exitosamente en:")
    print(base_dir)

if __name__ == "__main__":
    generar_dataset()
