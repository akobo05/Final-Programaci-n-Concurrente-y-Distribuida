import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Random;

public class NodoLider {
    private static final int PORT_CLIENT_SERVER = 5000;
    private static final int PORT_WEB = 8080;
    private static final String WORKER_HOST = "localhost";

    // Protocol Commands
    private static final byte CMD_HEARTBEAT = 0x01;
    private static final byte CMD_DISTRIBUTE_CHUNK = 0x03;
    private static final byte CMD_TRAIN_RESULT = 0x04;

    // Worker Connection Wrapper
    static class WorkerConnection {
        Socket socket;
        DataOutputStream out;
        DataInputStream in;
        int port;
        String host;

        public WorkerConnection(String host, int port) throws IOException {
            this.port = port;
            this.host = host;
            this.socket = new Socket(host, port);
            this.out = new DataOutputStream(socket.getOutputStream());
            this.in = new DataInputStream(socket.getInputStream());
        }
    }

    // Shared State
    private static List<WorkerConnection> workers = new CopyOnWriteArrayList<>();
    private static final Object workersLock = new Object();
    private static AtomicInteger trainedModels = new AtomicInteger(0);

    public static void main(String[] args) {
        System.out.println("Starting Leader Node...");

        // Parse arguments for worker ports
        List<Integer> workerPorts = new ArrayList<>();
        if (args.length > 0) {
            for (String arg : args) {
                try {
                    workerPorts.add(Integer.parseInt(arg));
                } catch (NumberFormatException e) {
                    System.err.println("Invalid port: " + arg);
                }
            }
        } else {
            // Default fallback if no args provided
            workerPorts.add(5001);
        }

        // 0. Load State
        loadState();

        // 1. Connect to Workers
        connectToWorkers(workerPorts);

        // 2. Start RAFT Heartbeat Thread
        new Thread(new RaftController()).start();

        // 3. Start Web Monitor Thread
        new Thread(new WebMonitor()).start();

        // 4. Start Client Server (Main Thread)
        startClientServer();
    }

    private static void connectToWorkers(List<Integer> ports) {
        for (int port : ports) {
            boolean connected = false;
            // Retry loop for each worker
            while (!connected) {
                try {
                    WorkerConnection worker = new WorkerConnection(WORKER_HOST, port);
                    workers.add(worker);
                    System.out.println("Connected to Worker at " + WORKER_HOST + ":" + port);
                    connected = true;
                } catch (IOException e) {
                    System.out.println("Waiting for Worker on port " + port + "... (" + e.getMessage() + ")");
                    try {
                        Thread.sleep(2000);
                    } catch (InterruptedException ex) {
                    }
                }
            }
        }
    }

    // RAFT Heartbeat Controller
    static class RaftController implements Runnable {
        @Override
        public void run() {
            while (true) {
                try {
                    synchronized (workersLock) {
                        for (WorkerConnection worker : workers) {
                            try {
                                worker.out.writeByte(CMD_HEARTBEAT);
                                worker.out.writeInt(0); // Length 0
                                worker.out.flush();
                            } catch (IOException e) {
                                System.err.println("Error sending heartbeat to worker " + worker.port + ": " + e.getMessage());
                                // Potential reconnection logic or removal
                            }
                        }
                    }
                    Thread.sleep(5000);
                } catch (Exception e) {
                    System.err.println("Error in RaftController: " + e.getMessage());
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

                    // Parse: [NumSamples (4)] [InputSize (4)] [Inputs...] [Targets...]
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

                    System.out
                            .println("[CLIENT] Training Request: " + numSamples + " images (size " + inputSize + ").");

                    // Initial Random Weights
                    double[] w = new double[inputSize];
                    Random rand = new Random();
                    for(int j=0; j<inputSize; j++) {
                        w[j] = (rand.nextDouble() * 0.02) - 0.01; // -0.01 to 0.01
                    }
                    double b = (rand.nextDouble() * 0.02) - 0.01;

                    // Training Loop (50 Epochs)
                    System.out.println("Starting 50 epochs of distributed training...");

                    for (int epoch = 0; epoch < 50; epoch++) {
                        // Split Data among Leader + Workers
                        int totalNodes = 1 + workers.size();
                        int chunkSize = numSamples / totalNodes;

                        // --- 1. Leader Processing (Local) ---
                        int startIdx = 0;
                        int endIdx = chunkSize + (numSamples % totalNodes); // Give remainder to leader

                        double[][] localInputs = Arrays.copyOfRange(inputs, startIdx, endIdx);
                        double[] localTargets = Arrays.copyOfRange(targets, startIdx, endIdx);

                        // Pass current weights to local compute
                        double[] localGrads = TrainingCore.computeGradients(localInputs, localTargets, inputSize, w, b);

                        startIdx = endIdx;

                        // --- 2. Worker Processing (Distributed) ---
                        List<double[]> allWorkerGrads = new ArrayList<>();

                        if (!workers.isEmpty()) {
                            synchronized (workersLock) {
                                for (WorkerConnection worker : workers) {
                                    int wStart = startIdx;
                                    int wEnd = startIdx + chunkSize;
                                    startIdx = wEnd;

                                    int wCount = wEnd - wStart;
                                    if (wCount <= 0) continue;

                                    try {
                                        worker.out.writeByte(CMD_DISTRIBUTE_CHUNK);
                                        // Payload: [NumSamples][InputSize][Weights][Bias][Inputs][Targets]
                                        int payloadSize = 4 + 4 + (inputSize * 8) + 8 + (wCount * inputSize * 8) + (wCount * 8);
                                        worker.out.writeInt(payloadSize);
                                        worker.out.writeInt(wCount);
                                        worker.out.writeInt(inputSize);

                                        // Send Weights & Bias
                                        for(double val : w) worker.out.writeDouble(val);
                                        worker.out.writeDouble(b);

                                        // Send Data
                                        for (int i = wStart; i < wEnd; i++) {
                                            for (int j = 0; j < inputSize; j++)
                                                worker.out.writeDouble(inputs[i][j]);
                                        }
                                        for (int i = wStart; i < wEnd; i++)
                                            worker.out.writeDouble(targets[i]);
                                        worker.out.flush();
                                    } catch (IOException e) {
                                        System.err.println("Error sending chunk to worker " + worker.port);
                                    }
                                }

                                // Receive results
                                for (WorkerConnection worker : workers) {
                                    try {
                                        byte respHeader = worker.in.readByte();
                                        if (respHeader == CMD_TRAIN_RESULT) {
                                            int respLen = worker.in.readInt();
                                            double[] wGrads = new double[inputSize + 1];
                                            for (int j = 0; j < inputSize; j++)
                                                wGrads[j] = worker.in.readDouble();
                                            wGrads[inputSize] = worker.in.readDouble(); // Bias
                                            allWorkerGrads.add(wGrads);
                                        }
                                    } catch (IOException e) {
                                        System.err.println("Error receiving from worker " + worker.port);
                                    }
                                }
                            }
                        }

                        // --- 3. Aggregate Gradients ---
                        double[] avg_dw = new double[inputSize];
                        double avg_db = 0.0;

                        // Sum local
                        for (int j = 0; j < inputSize; j++) avg_dw[j] += localGrads[j];
                        avg_db += localGrads[inputSize];

                        // Sum workers
                        for (double[] wg : allWorkerGrads) {
                            for (int j = 0; j < inputSize; j++) avg_dw[j] += wg[j];
                            avg_db += wg[inputSize];
                        }

                        // Average
                        int contributingNodes = 1 + allWorkerGrads.size();
                        for (int j = 0; j < inputSize; j++) avg_dw[j] /= contributingNodes;
                        avg_db /= contributingNodes;

                        // Update Weights (Learning Rate 0.1 for Logistic Regression)
                        // Gradient Descent: w = w - lr * grad
                        double learningRate = 0.1;
                        for (int j = 0; j < inputSize; j++)
                            w[j] = w[j] - (learningRate * avg_dw[j]);
                        b = b - (learningRate * avg_db);
                    }

                    System.out.println("Training Complete (50 Epochs).");

                    // Save Model
                    int modelId = trainedModels.incrementAndGet();
                    Model newModel = new Model(w, b);
                    modelStore.put(modelId, newModel);
                    modelNames.put(modelId, "Model " + modelId);

                    // Save State
                    saveState();

                    // RAFT Commit to ALL Workers
                    synchronized (workersLock) {
                        for (WorkerConnection worker : workers) {
                            try {
                                worker.out.writeByte(0x07); // COMMIT_ENTRY
                                // Payload: [ModelID][InputSize][w...][b]
                                worker.out.writeInt(4 + 4 + (inputSize * 8) + 8);
                                worker.out.writeInt(modelId);
                                worker.out.writeInt(inputSize);
                                for (double val : w)
                                    worker.out.writeDouble(val);
                                worker.out.writeDouble(b);
                                worker.out.flush();
                            } catch (IOException e) {
                                System.err.println("Error committing to worker " + worker.port);
                            }
                        }
                    }

                    out.writeUTF("Training Complete. Model ID: " + modelId);
                } else if (header == 0x06) { // PREDICT
                    int length = in.readInt();
                    byte[] payload = new byte[length];
                    in.readFully(payload);
                    ByteBuffer bb = ByteBuffer.wrap(payload);

                    int modelId = bb.getInt();
                    int inputSize = (length - 4) / 8; // Remaining bytes are doubles

                    double[] inputVal = new double[inputSize];
                    for (int i = 0; i < inputSize; i++)
                        inputVal[i] = bb.getDouble();

                    Model m = modelStore.get(modelId);
                    if (m != null) {
                        if (m.w.length != inputSize) {
                            out.writeUTF("Error: Input size mismatch.");
                        } else {
                            double z = m.b;
                            for (int i = 0; i < inputSize; i++)
                                z += m.w[i] * inputVal[i];
                            double probability = 1.0 / (1.0 + Math.exp(-z));
                            String label = (probability > 0.5) ? "Dígito 1" : "Dígito 0";
                            out.writeUTF(label + " (Prob: " + String.format("%.2f", probability) + ")");
                        }
                    } else {
                        out.writeUTF("Error: Model " + modelId + " not found.");
                    }
                } else if (header == 0x08) { // GET_MODELS
                    int count = modelStore.size();
                    out.writeInt(count);
                    for (Integer id : modelStore.keySet()) {
                        out.writeInt(id);
                        out.writeUTF(modelNames.getOrDefault(id, "Model " + id));
                    }
                } else if (header == 0x09) { // UPDATE_MODEL_NAME
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
                    synchronized(workersLock) {
                        int count = workers.size();
                        out.writeInt(count);
                        for (WorkerConnection worker : workers) {
                            String host = worker.host;
                            if (host.equals("0:0:0:0:0:0:0:1")) host = "localhost";
                            out.writeUTF(host);
                            out.writeInt(worker.port);
                        }
                    }
                } else if (header == 0x11) { // KILL_WORKER
                    int portToKill = in.readInt();
                    boolean found = false;
                    WorkerConnection target = null;

                    synchronized(workersLock) {
                        for (WorkerConnection worker : workers) {
                            if (worker.port == portToKill) {
                                target = worker;
                                break;
                            }
                        }
                        if (target != null) {
                            try {
                                target.socket.close();
                            } catch (IOException e) {
                                // Ignore
                            }
                            workers.remove(target);
                            found = true;
                        }
                    }

                    if (found) {
                        out.writeUTF("Worker on port " + portToKill + " disconnected.");
                        System.out.println("Worker on port " + portToKill + " killed by client request.");
                    } else {
                        out.writeUTF("Worker not found.");
                    }
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
            System.out.println("State saved to disk.");
        } catch (IOException e) {
            System.err.println("Error saving state: " + e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private static void loadState() {
        File file = new File("estado_sistema.dat");
        if (!file.exists()) {
            System.out.println("No saved state found. Starting fresh.");
            return;
        }
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            trainedModels = (AtomicInteger) ois.readObject();
            modelStore = (java.util.Map<Integer, Model>) ois.readObject();
            modelNames = (java.util.Map<Integer, String>) ois.readObject();
            System.out.println("State restored. Models count: " + trainedModels.get());
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error loading state (starting fresh): " + e.getMessage());
            trainedModels = new AtomicInteger(0);
            modelStore = new java.util.concurrent.ConcurrentHashMap<>();
            modelNames = new java.util.concurrent.ConcurrentHashMap<>();
        }
    }

    static class TrainingCore {
        private static double sigmoid(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        public static double[] computeGradients(double[][] inputs, double[] targets, int inputSize, double[] w, double b) {
            double[] grad_w = new double[inputSize];
            double grad_b = 0.0;

            for (int i = 0; i < inputs.length; i++) {
                double z = b;
                for (int j = 0; j < inputSize; j++)
                    z += w[j] * inputs[i][j];

                double pred = sigmoid(z);
                double error = targets[i] - pred;

                // Gradient Descent for Logistic Regression (Cross-Entropy)
                // dL/dw = (pred - y) * x
                // Here error = y - pred, so dL/dw = -error * x
                // Maintaining the -2 factor to align with previous linear regression style
                // and the worker implementation, effectively increasing learning rate by 2.
                for (int j = 0; j < inputSize; j++)
                    grad_w[j] += -2 * inputs[i][j] * error;
                grad_b += -2 * error;
            }

            if (inputs.length > 0) {
                for (int j = 0; j < inputSize; j++)
                    grad_w[j] /= inputs.length;
                grad_b /= inputs.length;
            }

            double[] result = Arrays.copyOf(grad_w, inputSize + 1);
            result[inputSize] = grad_b;
            return result;
        }
    }
}
