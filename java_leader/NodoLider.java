import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Random;

public class NodoLider {
    private static final int PORT_CLIENT_SERVER = 5000;
    private static final int PORT_WORKER_SERVER = 9000;
    private static final int PORT_WEB = 8080;

    // Protocol Commands
    private static final byte CMD_HEARTBEAT = 0x01;
    private static final byte CMD_DISTRIBUTE_CHUNK = 0x03;
    private static final byte CMD_TRAIN_RESULT = 0x04;

    // Worker Connection Wrapper
    static class WorkerConnection {
        Socket socket;
        DataOutputStream out;
        DataInputStream in;
        int remotePort; // Port reported by Worker for ID
        String remoteIp;

        public WorkerConnection(Socket socket) throws IOException {
            this.socket = socket;
            this.out = new DataOutputStream(socket.getOutputStream());
            this.in = new DataInputStream(socket.getInputStream());
            this.remoteIp = socket.getInetAddress().getHostAddress();

            // Handshake to get Identity
            // Worker sends 4-byte Port ID
            // Set timeout for handshake
            socket.setSoTimeout(5000);
            try {
                this.remotePort = in.readInt();
            } catch (Exception e) {
                this.remotePort = -1;
            }
            socket.setSoTimeout(0); // Reset
        }
    }

    // Shared State
    private static CopyOnWriteArrayList<WorkerConnection> workers = new CopyOnWriteArrayList<>();
    private static AtomicInteger trainedModels = new AtomicInteger(0);

    public static void main(String[] args) {
        System.out.println("Starting Leader Node...");

        // 0. Load State
        loadState();

        // 1. Start Worker Server (Accepts Workers)
        new Thread(new WorkerServer()).start();

        // 2. Start RAFT Heartbeat Thread
        new Thread(new RaftController()).start();

        // 3. Start Web Monitor Thread
        new Thread(new WebMonitor()).start();

        // 4. Start Client Server (Main Thread)
        startClientServer();
    }

    // Worker Server Logic
    static class WorkerServer implements Runnable {
        @Override
        public void run() {
            try (ServerSocket server = new ServerSocket(PORT_WORKER_SERVER)) {
                System.out.println("Worker Server listening on port " + PORT_WORKER_SERVER);
                while (true) {
                    try {
                        Socket socket = server.accept();
                        WorkerConnection wc = new WorkerConnection(socket);
                        workers.add(wc);
                        System.out.println("Worker Connected: " + wc.remoteIp + " (ID: " + wc.remotePort + ")");
                    } catch (IOException e) {
                        System.err.println("Error accepting worker: " + e.getMessage());
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // RAFT Heartbeat Controller
    static class RaftController implements Runnable {
        @Override
        public void run() {
            while (true) {
                try {
                    for (WorkerConnection worker : workers) {
                        try {
                            synchronized (worker.out) {
                                worker.out.writeByte(CMD_HEARTBEAT);
                                worker.out.writeInt(0); // Length 0
                                worker.out.flush();
                            }
                        } catch (IOException e) {
                            System.err.println("Worker " + worker.remotePort + " failed heartbeat. Removing.");
                            workers.remove(worker);
                            try { worker.socket.close(); } catch (Exception ex) {}
                        }
                    }
                    Thread.sleep(5000);
                } catch (Exception e) {
                    // Ignore
                }
            }
        }
    }

    // Web Monitor
    static class WebMonitor implements Runnable {
        @Override
        public void run() {
            try (ServerSocket server = new ServerSocket(PORT_WEB)) {
                System.out.println("Web Monitor listening on port " + PORT_WEB);
                while (true) {
                    try (Socket client = server.accept()) {
                        String response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n" +
                                "<html><body><h1>Leader Node (Java)</h1>" +
                                "<p>Status: <b>LEADER</b></p>" +
                                "<p>Workers Connected: " + workers.size() + "</p>" +
                                "<p>Models Trained: " + trainedModels.get() + "</p>" +
                                "</body></html>";
                        client.getOutputStream().write(response.getBytes());
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // Client Server & Training Logic
    private static void startClientServer() {
        try (ServerSocket server = new ServerSocket(PORT_CLIENT_SERVER)) {
            System.out.println("Client Server listening on port " + PORT_CLIENT_SERVER);
            while (true) {
                Socket client = server.accept();
                new Thread(new ClientHandler(client)).start();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Model Storage
    static class Model implements Serializable {
        private static final long serialVersionUID = 1L;
        double[] w;
        double b;

        public Model(double[] w, double b) {
            this.w = w;
            this.b = b;
        }
    }

    private static java.util.Map<Integer, Model> modelStore = new java.util.concurrent.ConcurrentHashMap<>();
    private static java.util.Map<Integer, String> modelNames = new java.util.concurrent.ConcurrentHashMap<>();

    static class Chunk {
        double[][] inputs;
        double[] targets;
        int startIdx;

        public Chunk(double[][] inputs, double[] targets, int startIdx) {
            this.inputs = inputs;
            this.targets = targets;
            this.startIdx = startIdx;
        }
    }

    static class ClientHandler implements Runnable {
        private Socket client;

        public ClientHandler(Socket client) {
            this.client = client;
        }

        @Override
        public void run() {
            try (DataInputStream in = new DataInputStream(client.getInputStream());
                    DataOutputStream out = new DataOutputStream(client.getOutputStream())) {

                byte header = in.readByte();

                if (header == 0x02) { // START_TRAIN
                    int length = in.readInt();
                    byte[] payload = new byte[length];
                    in.readFully(payload);
                    ByteBuffer bb = ByteBuffer.wrap(payload);

                    int numSamples = bb.getInt();
                    int inputSize = bb.getInt();

                    double[][] inputs = new double[numSamples][inputSize];
                    double[] targets = new double[numSamples];

                    for (int i = 0; i < numSamples; i++) {
                        for (int j = 0; j < inputSize; j++) {
                            inputs[i][j] = bb.getDouble();
                        }
                    }
                    for (int i = 0; i < numSamples; i++)
                        targets[i] = bb.getDouble();

                    System.out.println("[CLIENT] Training Request: " + numSamples + " images.");

                    // Initial Random Weights (-0.01 to 0.01)
                    double[] w = new double[inputSize];
                    Random rand = new Random();
                    for(int j=0; j<inputSize; j++) {
                        w[j] = (rand.nextDouble() * 0.02) - 0.01;
                    }
                    double b = (rand.nextDouble() * 0.02) - 0.01;

                    System.out.println("Starting 50 epochs of distributed training...");

                    for (int epoch = 0; epoch < 50; epoch++) {
                        // 1. Create Task Queue
                        ConcurrentLinkedQueue<Chunk> taskQueue = new ConcurrentLinkedQueue<>();

                        // Decide chunk size based on CURRENT workers (soft target) or fixed size
                        // Better to use fixed reasonable size or divide by current workers.
                        int currentWorkerCount = workers.size();
                        int splits = Math.max(1, currentWorkerCount); // at least 1 split
                        int chunkSize = numSamples / splits;
                        if (chunkSize < 1) chunkSize = 1;

                        int start = 0;
                        while (start < numSamples) {
                            int end = Math.min(start + chunkSize, numSamples);
                            double[][] chunkInputs = Arrays.copyOfRange(inputs, start, end);
                            double[] chunkTargets = Arrays.copyOfRange(targets, start, end);
                            taskQueue.add(new Chunk(chunkInputs, chunkTargets, start));
                            start = end;
                        }

                        List<double[]> gradients = new ArrayList<>();

                        // Process Queue
                        while (!taskQueue.isEmpty()) {
                            // Find available worker
                            WorkerConnection assignedWorker = null;
                            for (WorkerConnection wc : workers) {
                                assignedWorker = wc;
                                break;
                            }

                            if (workers.isEmpty()) {
                                System.out.println("No workers available! Waiting...");
                                try { Thread.sleep(1000); } catch (Exception e) {}
                                continue;
                            }

                            // Let's implement the "Master-Worker" pattern correctly.
                            // We have N workers. We have M chunks.
                            // Create N threads (one per worker).
                            // Each thread pulls from `taskQueue`.

                            List<Thread> threads = new ArrayList<>();
                            List<WorkerConnection> snapshotWorkers = new ArrayList<>(workers);

                            // Collections.synchronizedList for gradients
                            List<double[]> threadSafeGradients = java.util.Collections.synchronizedList(gradients);

                            final double bCopy = b;

                            for (WorkerConnection wc : snapshotWorkers) {
                                Thread t = new Thread(() -> {
                                    while (true) {
                                        Chunk c = taskQueue.poll();
                                        if (c == null) break; // No more work

                                        try {
                                            synchronized (wc.out) { // Lock socket
                                                wc.out.writeByte(CMD_DISTRIBUTE_CHUNK);
                                                // Payload
                                                int payloadSize = 4 + 4 + (inputSize * 8) + 8 + (c.inputs.length * inputSize * 8) + (c.inputs.length * 8);
                                                wc.out.writeInt(payloadSize);
                                                wc.out.writeInt(c.inputs.length);
                                                wc.out.writeInt(inputSize);
                                                // Weights
                                                for(double val : w) wc.out.writeDouble(val);
                                                wc.out.writeDouble(bCopy);
                                                // Data
                                                for (int i = 0; i < c.inputs.length; i++) {
                                                    for (int j = 0; j < inputSize; j++) wc.out.writeDouble(c.inputs[i][j]);
                                                }
                                                for (int i = 0; i < c.targets.length; i++) wc.out.writeDouble(c.targets[i]);
                                                wc.out.flush();

                                                // Read Result
                                                byte respHeader = wc.in.readByte();
                                                if (respHeader == CMD_TRAIN_RESULT) {
                                                    int respLen = wc.in.readInt();
                                                    double[] wGrads = new double[inputSize + 1];
                                                    for (int j = 0; j < inputSize; j++) wGrads[j] = wc.in.readDouble();
                                                    wGrads[inputSize] = wc.in.readDouble(); // Bias
                                                    threadSafeGradients.add(wGrads);
                                                } else {
                                                    throw new IOException("Unexpected header: " + respHeader);
                                                }
                                            }
                                        } catch (IOException e) {
                                            System.out.println("Worker " + wc.remotePort + " failed. Reassigning task...");
                                            taskQueue.add(c); // Reassign
                                            workers.remove(wc);
                                            try { wc.socket.close(); } catch(Exception ex) {}
                                            break; // Worker dead, thread exit
                                        }
                                    }
                                });
                                t.start();
                                threads.add(t);
                            }

                            for (Thread t : threads) {
                                try { t.join(); } catch (InterruptedException e) {}
                            }

                            // If queue still has items (all workers died?), we loop again
                            if (!taskQueue.isEmpty()) {
                                if (workers.isEmpty()) {
                                    System.out.println("All workers dead. Pausing...");
                                    try { Thread.sleep(2000); } catch (Exception e) {}
                                }
                                continue; // Retry remaining chunks
                            } else {
                                break; // Epoch done
                            }
                        }

                        // Aggregate
                        if (gradients.isEmpty()) continue;

                        double[] avg_dw = new double[inputSize];
                        double avg_db = 0.0;
                        for (double[] g : gradients) {
                            for (int j = 0; j < inputSize; j++) avg_dw[j] += g[j];
                            avg_db += g[inputSize];
                        }
                        int N = gradients.size();
                        for (int j = 0; j < inputSize; j++) avg_dw[j] /= N;
                        avg_db /= N;

                        // Update
                        double lr = 0.1;
                        for (int j = 0; j < inputSize; j++) w[j] -= lr * avg_dw[j];
                        b -= lr * avg_db;

                        // Progress Update to Client
                        if (epoch % 5 == 0 || epoch == 49) {
                            out.writeUTF("Progress: Epoch " + (epoch + 1) + "/50 completed.");
                            out.flush();
                        }
                    }

                    System.out.println("Training Complete.");

                    // Save Model
                    int modelId = trainedModels.incrementAndGet();
                    Model newModel = new Model(w, b);
                    modelStore.put(modelId, newModel);
                    modelNames.put(modelId, "Model " + modelId);
                    saveState();

                    // Commit to Workers
                    for (WorkerConnection worker : workers) {
                        try {
                            synchronized(worker.out) {
                                worker.out.writeByte(0x07); // COMMIT
                                worker.out.writeInt(4 + 4 + (inputSize * 8) + 8);
                                worker.out.writeInt(modelId);
                                worker.out.writeInt(inputSize);
                                for (double val : w) worker.out.writeDouble(val);
                                worker.out.writeDouble(b);
                                worker.out.flush();
                            }
                        } catch (Exception e) {
                            // worker might have died
                        }
                    }

                    out.writeUTF("Training Complete. Model ID: " + modelId);

                } else if (header == 0x06) { // PREDICT
                    // ... Same as before ...
                    int length = in.readInt();
                    byte[] payload = new byte[length];
                    in.readFully(payload);
                    ByteBuffer bb = ByteBuffer.wrap(payload);
                    int modelId = bb.getInt();
                    int inputSize = (length - 4) / 8;
                    double[] inputVal = new double[inputSize];
                    for (int i = 0; i < inputSize; i++) inputVal[i] = bb.getDouble();

                    Model m = modelStore.get(modelId);
                    if (m != null) {
                        double z = m.b;
                        for (int i = 0; i < inputSize; i++) z += m.w[i] * inputVal[i];
                        double probability = 1.0 / (1.0 + Math.exp(-z));
                        String label = (probability > 0.5) ? "Dígito 1" : "Dígito 0";
                        out.writeUTF(label + " (Prob: " + String.format("%.2f", probability) + ")");
                    } else {
                        out.writeUTF("Error: Model not found.");
                    }

                } else if (header == 0x08) { // GET_MODELS
                    int count = modelStore.size();
                    out.writeInt(count);
                    for (Integer id : modelStore.keySet()) {
                        out.writeInt(id);
                        out.writeUTF(modelNames.getOrDefault(id, "Model " + id));
                    }
                } else if (header == 0x09) { // UPDATE_NAME
                    int length = in.readInt();
                    byte[] payload = new byte[length];
                    in.readFully(payload);
                    ByteBuffer bb = ByteBuffer.wrap(payload);
                    int modelId = bb.getInt();
                    byte[] nameBytes = new byte[length - 4];
                    bb.get(nameBytes);
                    String name = new String(nameBytes, java.nio.charset.StandardCharsets.UTF_8);
                    if (modelStore.containsKey(modelId)) {
                        modelNames.put(modelId, name);
                        out.writeUTF("Name updated.");
                    } else {
                        out.writeUTF("Model not found.");
                    }
                } else if (header == 0x10) { // LIST_WORKERS
                    int count = workers.size();
                    out.writeInt(count);
                    for (WorkerConnection worker : workers) {
                        out.writeUTF(worker.remoteIp);
                        out.writeInt(worker.remotePort);
                    }
                } else if (header == 0x11) { // KILL_WORKER
                    int portToKill = in.readInt();
                    boolean found = false;
                    for (WorkerConnection worker : workers) {
                        if (worker.remotePort == portToKill) {
                            try { worker.socket.close(); } catch (Exception e) {}
                            workers.remove(worker);
                            found = true;
                            break;
                        }
                    }
                    if (found) out.writeUTF("Worker killed.");
                    else out.writeUTF("Worker not found.");
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static synchronized void saveState() {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("estado_sistema.dat"))) {
            oos.writeObject(trainedModels);
            oos.writeObject(modelStore);
            oos.writeObject(modelNames);
        } catch (IOException e) {
            System.err.println("Error saving state: " + e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private static void loadState() {
        File file = new File("estado_sistema.dat");
        if (!file.exists()) return;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            trainedModels = (AtomicInteger) ois.readObject();
            modelStore = (java.util.Map<Integer, Model>) ois.readObject();
            modelNames = (java.util.Map<Integer, String>) ois.readObject();
        } catch (Exception e) {
            trainedModels = new AtomicInteger(0);
            modelStore = new java.util.concurrent.ConcurrentHashMap<>();
            modelNames = new java.util.concurrent.ConcurrentHashMap<>();
        }
    }
}
