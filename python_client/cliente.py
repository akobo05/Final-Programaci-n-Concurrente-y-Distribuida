import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, ttk
import socket
import struct
import random
import threading
import os

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

NODES = [('localhost', 5000), ('localhost', 5001)]
IMG_SIZE = 32 # 32x32 = 1024 pixels

class DistributedClient:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Entrenamiento Distribuido")
        self.root.geometry("600x550")

        self.inputs = []
        self.targets = []
        self.model_list = [] # List of tuples (id, name)

        # UI Elements
        self.label_title = tk.Label(root, text="Sistema de Entrenamiento Distribuido", font=("Arial", 16, "bold"))
        self.label_title.pack(pady=10)

        # Failover Visual Indicator
        self.lbl_conn_info = tk.Label(root, text="Desconectado", fg="gray", font=("Arial", 10))
        self.lbl_conn_info.pack(pady=2)

        if not HAS_PIL:
            tk.Label(root, text="ADVERTENCIA: Librería 'Pillow' no encontrada. Instala con 'pip install Pillow'", fg="red").pack()

        # Training Section
        self.frame_train = tk.LabelFrame(root, text="Entrenamiento (Imágenes Reales)", padx=10, pady=10)
        self.frame_train.pack(pady=5, fill="x", padx=10)

        self.btn_load = tk.Button(self.frame_train, text="Seleccionar Carpeta Dataset", command=self.load_dataset_folder)
        self.btn_load.pack(pady=5)
        
        self.lbl_dataset_info = tk.Label(self.frame_train, text="No se ha cargado dataset.")
        self.lbl_dataset_info.pack(pady=5)

        self.btn_train = tk.Button(self.frame_train, text="Iniciar Entrenamiento", command=self.start_training_thread, state=tk.DISABLED)
        self.btn_train.pack(pady=5)

        # Cluster Management Section
        self.frame_cluster = tk.LabelFrame(root, text="Gestión del Cluster", padx=10, pady=10)
        self.frame_cluster.pack(pady=5, fill="x", padx=10)

        # Treeview
        columns = ("ip", "port")
        self.tree = ttk.Treeview(self.frame_cluster, columns=columns, show='headings', height=3)
        self.tree.heading("ip", text="IP")
        self.tree.heading("port", text="Puerto")
        self.tree.column("ip", width=100)
        self.tree.column("port", width=80)
        self.tree.pack(side=tk.LEFT, padx=5)

        self.btn_kill = tk.Button(self.frame_cluster, text="Simular Caída", command=self.kill_worker)
        self.btn_kill.pack(side=tk.LEFT, padx=5)

        # Testing Section
        self.frame_test = tk.LabelFrame(root, text="Testeo / Inferencia", padx=10, pady=10)
        self.frame_test.pack(pady=5, fill="x", padx=10)

        tk.Label(self.frame_test, text="Modelo:").pack(side=tk.LEFT)
        
        self.combo_models = ttk.Combobox(self.frame_test, width=20, state="readonly")
        self.combo_models.pack(side=tk.LEFT, padx=5)
        
        self.btn_refresh = tk.Button(self.frame_test, text="↻", command=self.fetch_models_thread)
        self.btn_refresh.pack(side=tk.LEFT, padx=2)

        self.btn_select_img = tk.Button(self.frame_test, text="Cargar Imagen...", command=self.select_inference_image)
        self.btn_select_img.pack(side=tk.LEFT, padx=10)

        self.log_text = tk.Text(root, height=10, width=70)
        self.log_text.pack(pady=10)
        
        # Initial fetch
        self.fetch_models_thread()
        # Start cluster monitor loop
        self.monitor_cluster_loop()

    def monitor_cluster_loop(self):
        threading.Thread(target=self.fetch_workers).start()
        self.root.after(3000, self.monitor_cluster_loop)

    def fetch_workers(self):
        try:
            s = self.get_active_leader_socket()
            with s:
                # Protocol: [0x10] (No payload)
                s.sendall(b'\x10')

                count_bytes = s.recv(4)
                if not count_bytes: return
                count = struct.unpack('>I', count_bytes)[0]

                workers = []
                for _ in range(count):
                    # Read UTF host
                    len_bytes = s.recv(2)
                    host_len = struct.unpack('>H', len_bytes)[0]
                    host_bytes = s.recv(host_len)
                    host = host_bytes.decode('utf-8')

                    # Read port
                    port_bytes = s.recv(4)
                    port = struct.unpack('>I', port_bytes)[0]

                    workers.append((host, port))

                self.root.after(0, lambda: self.update_cluster_ui(workers))
        except:
            pass # Silent fail if leader down

    def update_cluster_ui(self, workers):
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)

        for w in workers:
            self.tree.insert('', tk.END, values=w)

    def kill_worker(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showerror("Error", "Selecciona un worker de la lista.")
            return

        item = self.tree.item(selection[0])
        values = item['values']
        # values is a list/tuple: ['ip', port]
        # Tkinter treeview values are often strings even if inserted as int
        port = int(values[1])

        if messagebox.askyesno("Confirmar", f"¿Seguro que deseas desconectar el worker en puerto {port}?"):
            threading.Thread(target=self.send_kill_command, args=(port,)).start()

    def send_kill_command(self, port):
        try:
            s = self.get_active_leader_socket()
            with s:
                # Protocol: [0x11] [Port]
                s.sendall(b'\x11')
                s.sendall(struct.pack('>I', port))

                resp_len_bytes = s.recv(2)
                resp_len = struct.unpack('>H', resp_len_bytes)[0]
                resp_bytes = s.recv(resp_len)
                msg = resp_bytes.decode('utf-8')

                self.root.after(0, lambda: self.log(f"Respuesta Cluster: {msg}"))
                self.fetch_workers() # Refresh list immediately
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Error gestionando cluster: {e}"))

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def process_image(self, path):
        if not HAS_PIL: return None
        try:
            img = Image.open(path).convert('L') # Grayscale
            img = img.resize((IMG_SIZE, IMG_SIZE))
            # Normalize 0-1
            pixels = list(img.getdata())
            return [p/255.0 for p in pixels]
        except Exception as e:
            self.log(f"Error procesando {path}: {e}")
            return None

    def load_dataset_folder(self):
        if not HAS_PIL:
            messagebox.showerror("Error", "Necesitas Pillow para cargar imágenes.")
            return

        folder_path = filedialog.askdirectory(title="Seleccionar Carpeta con Dataset")
        if not folder_path: return

        self.inputs = []
        self.targets = []
        
        classes = ['0', '1']
        found_classes = False
        
        for cls in classes:
            cls_path = os.path.join(folder_path, cls)
            if os.path.isdir(cls_path):
                found_classes = True
                self.log(f"Cargando clase {cls}...")
                for fname in os.listdir(cls_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        vec = self.process_image(os.path.join(cls_path, fname))
                        if vec:
                            self.inputs.append(vec)
                            self.targets.append(float(cls))
        
        if not found_classes:
            self.log("No se encontraron subcarpetas '0' y '1'. Cargando todo como Clase 0.")
            for fname in os.listdir(folder_path):
                 if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        vec = self.process_image(os.path.join(folder_path, fname))
                        if vec:
                            self.inputs.append(vec)
                            self.targets.append(0.0)

        if self.inputs:
            self.lbl_dataset_info.config(text=f"Dataset: {len(self.inputs)} imágenes (Tamaño {IMG_SIZE}x{IMG_SIZE})")
            self.log(f"Cargadas {len(self.inputs)} imágenes.")
            self.btn_train.config(state=tk.NORMAL)
        else:
            self.log("No se encontraron imágenes válidas.")

    def get_active_leader_socket(self):
        """
        Intenta conectar a los nodos definidos en NODES uno por uno.
        Retorna el socket conectado si tiene éxito.
        """
        for host, port in NODES:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0) # 1 segundo timeout para descubrimiento rápido
                s.connect((host, port))
                s.settimeout(None) # Quitar timeout para operaciones normales

                # Actualizar UI (thread safe)
                self.root.after(0, lambda h=host, p=port: self.lbl_conn_info.config(
                    text=f"Conectado a Líder en: {h}:{p}", fg="green"
                ))
                return s
            except OSError:
                s.close()

        # Si fallan todos
        self.root.after(0, lambda: self.lbl_conn_info.config(
            text="Error: No se pudo conectar a ningún nodo.", fg="red"
        ))
        raise Exception("No se pudo conectar a ningún nodo líder activo.")

    def start_training_thread(self):
        threading.Thread(target=self.start_training).start()

    def start_training(self):
        if not self.inputs:
            return

        self.btn_train.config(state=tk.DISABLED)
        self.log("Buscando líder y conectando...")

        try:
            s = self.get_active_leader_socket()
            with s:
                self.log("Conectado. Enviando datos...")

                # Protocol: [0x02] [Length] [NumSamples] [InputSize] [Inputs...] [Targets...]
                header = b'\x02'
                
                num_samples = len(self.inputs)
                input_size = IMG_SIZE * IMG_SIZE
                
                payload = struct.pack('>I', num_samples) + struct.pack('>I', input_size)
                for vec in self.inputs:
                    payload += b''.join([struct.pack('>d', x) for x in vec])
                payload += b''.join([struct.pack('>d', y) for y in self.targets])
                
                length = len(payload)
                
                s.sendall(header)
                s.sendall(struct.pack('>I', length))
                s.sendall(payload)

                self.log("Datos enviados. Esperando respuesta...")
                
                response_data = s.recv(1024)
                if len(response_data) > 2:
                    msg_len = struct.unpack('>H', response_data[:2])[0]
                    msg = response_data[2:2+msg_len].decode('utf-8')
                else:
                    msg = response_data.decode('utf-8', errors='ignore')

                self.log(f"Respuesta: {msg}")
                
                # Parse Model ID from response "Training Complete. Model ID: X"
                if "Model ID:" in msg:
                    try:
                        model_id = int(msg.split("Model ID:")[1].strip())
                        self.root.after(0, lambda: self.prompt_model_name(model_id))
                    except:
                        pass

        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.btn_train.config(state=tk.NORMAL)
            self.fetch_models_thread()

    def prompt_model_name(self, model_id):
        name = simpledialog.askstring("Nombre del Modelo", f"Entrenamiento finalizado (ID: {model_id}).\nAsigna un nombre:", initialvalue=f"Model {model_id}")
        if name:
            threading.Thread(target=self.update_model_name, args=(model_id, name)).start()

    def update_model_name(self, model_id, name):
        try:
            s = self.get_active_leader_socket()
            with s:
                # Protocol: [0x09] [Length] [ModelID] [Name]
                header = b'\x09'
                name_bytes = name.encode('utf-8')
                payload = struct.pack('>I', model_id) + name_bytes
                length = len(payload)
                
                s.sendall(header)
                s.sendall(struct.pack('>I', length))
                s.sendall(payload)
                
                resp = s.recv(1024) # "Name updated."
                self.log(f"Nombre actualizado: {name}")
                self.fetch_models_thread()
        except Exception as e:
            self.log(f"Error updating name: {e}")

    def fetch_models_thread(self):
        threading.Thread(target=self.fetch_models).start()

    def fetch_models(self):
        try:
            s = self.get_active_leader_socket()
            with s:
                # Protocol: [0x08] (No payload)
                s.sendall(b'\x08')
                
                count_bytes = s.recv(4)
                if not count_bytes: return
                count = struct.unpack('>I', count_bytes)[0]
                
                new_list = []
                for _ in range(count):
                    id_bytes = s.recv(4)
                    model_id = struct.unpack('>I', id_bytes)[0]
                    
                    # Read UTF string manually or use helper? Java writeUTF writes length (2 bytes) then bytes.
                    len_bytes = s.recv(2)
                    name_len = struct.unpack('>H', len_bytes)[0]
                    name_bytes = s.recv(name_len)
                    name = name_bytes.decode('utf-8')
                    
                    new_list.append((model_id, name))
                
                self.model_list = new_list
                self.root.after(0, self.update_combo_box)
                
        except Exception as e:
            # self.log(f"Error fetching models: {e}") # Don't spam log on startup if server down
            pass

    def update_combo_box(self):
        values = [f"{m[1]} (ID: {m[0]})" for m in self.model_list]
        self.combo_models['values'] = values
        if values:
            self.combo_models.current(len(values)-1) # Select last

    def select_inference_image(self):
        if not HAS_PIL:
            messagebox.showerror("Error", "Necesitas Pillow.")
            return
            
        selection = self.combo_models.get()
        if not selection:
            messagebox.showerror("Error", "Selecciona un modelo.")
            return
            
        # Extract ID from "Name (ID: X)"
        try:
            model_id = int(selection.split("(ID:")[1].strip()[:-1])
        except:
            messagebox.showerror("Error", "Formato de modelo inválido.")
            return

        file_path = filedialog.askopenfilename(title="Seleccionar Imagen")
        if not file_path: return
        
        vec = self.process_image(file_path)
        if vec:
            threading.Thread(target=self.send_prediction, args=(model_id, vec)).start()

    def send_prediction(self, model_id, val_vec):
        try:
            s = self.get_active_leader_socket()
            with s:
                # Protocol: [0x06] [Length] [ModelID] [InputVec...]
                header = b'\x06'
                payload = struct.pack('>I', model_id)
                payload += b''.join([struct.pack('>d', x) for x in val_vec])
                length = len(payload)

                s.sendall(header)
                s.sendall(struct.pack('>I', length))
                s.sendall(payload)
                
                response_data = s.recv(1024)
                if len(response_data) > 2:
                    msg_len = struct.unpack('>H', response_data[:2])[0]
                    msg = response_data[2:2+msg_len].decode('utf-8')
                else:
                    msg = response_data.decode('utf-8', errors='ignore')

                self.root.after(0, lambda: self.log(f"Predicción (M{model_id}): {msg}"))

        except Exception as e:
            self.root.after(0, lambda: self.log(f"Error Test: {e}"))

if __name__ == "__main__":
    root = tk.Tk()
    app = DistributedClient(root)
    root.mainloop()
