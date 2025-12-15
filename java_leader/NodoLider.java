import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

public class NodoLider {
    private static final int PORT_CLIENT_SERVER = 5000;
    private static final int PORT_WORKER_TARGET = 5001;
    private static final int PORT_WEB = 8080;
    private static final String WORKER_HOST = "localhost";

    // Protocol Commands
    private static final byte CMD_HEARTBEAT = 0x01;
    private static final byte CMD_DISTRIBUTE_CHUNK = 0x03;
    private static final byte CMD_TRAIN_RESULT = 0x04;

    // Shared State
    private static Socket workerSocket;
    private static DataOutputStream workerOut;
    private static DataInputStream workerIn;
    private static final Object workerLock = new Object();
    private static AtomicInteger trainedModels = new AtomicInteger(0);

    public static void main(String[] args) {
        System.out.println("Starting Leader Node...");

        // 1. Connect to Worker (Follower)
        connectToWorker();

        // 2. Start RAFT Heartbeat Thread
        new Thread(new RaftController()).start();

        // 3. Start Web Monitor Thread
        new Thread(new WebMonitor()).start();

        // 4. Start Client Server (Main Thread)
        startClientServer();
    }

    private static void connectToWorker() {
        while (true) {
            try {
                workerSocket = new Socket(WORKER_HOST, PORT_WORKER_TARGET);
                workerOut = new DataOutputStream(workerSocket.getOutputStream());
                workerIn = new DataInputStream(workerSocket.getInputStream());
                System.out.println("Connected to Worker at " + WORKER_HOST + ":" + PORT_WORKER_TARGET);
                break;
            } catch (IOException e) {
                System.out.println("Waiting for Worker... (" + e.getMessage() + ")");
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException ex) {
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
                    synchronized (workerLock) {
                        if (workerOut != null) {
                            workerOut.writeByte(CMD_HEARTBEAT);
                            workerOut.writeInt(0); // Length 0
                            workerOut.flush();
                            // System.out.println("[RAFT] Heartbeat sent.");
                        }
                    }
                    Thread.sleep(5000);
                } catch (Exception e) {
                    System.err.println("Error sending heartbeat: " + e.getMessage());
                    // Reconnect logic could go here
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
                                "<p>Workers Connected: 1</p>" +
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
    static class Model {
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

                    // Split Data
                    int mid = numSamples / 2;

                    // Local Processing
                    double[][] localInputs = Arrays.copyOfRange(inputs, 0, mid);
                    double[] localTargets = Arrays.copyOfRange(targets, 0, mid);
                    double[] localGrads = TrainingCore.computeGradients(localInputs, localTargets, inputSize); // [dw1,
                                                                                                               // dw2...,
                                                                                                               // db]

                    // Worker Processing
                    double[] workerGrads = new double[inputSize + 1];
                    int workerCount = numSamples - mid;

                    if (workerCount > 0) {
                        synchronized (workerLock) {
                            workerOut.writeByte(CMD_DISTRIBUTE_CHUNK);
                            // Payload: [NumSamples][InputSize][Inputs][Targets]
                            int payloadSize = 4 + 4 + (workerCount * inputSize * 8) + (workerCount * 8);
                            workerOut.writeInt(payloadSize);
                            workerOut.writeInt(workerCount);
                            workerOut.writeInt(inputSize);
                            for (int i = mid; i < numSamples; i++) {
                                for (int j = 0; j < inputSize; j++)
                                    workerOut.writeDouble(inputs[i][j]);
                            }
                            for (int i = mid; i < numSamples; i++)
                                workerOut.writeDouble(targets[i]);
                            workerOut.flush();
                        }

                        // Wait for Result
                        synchronized (workerLock) {
                            byte respHeader = workerIn.readByte();
                            if (respHeader == CMD_TRAIN_RESULT) {
                                int respLen = workerIn.readInt();
                                for (int j = 0; j < inputSize; j++)
                                    workerGrads[j] = workerIn.readDouble();
                                workerGrads[inputSize] = workerIn.readDouble(); // Bias
                                System.out.println("[LEADER] Worker Grads received.");
                            }
                        }
                    }

                    // Aggregate Gradients
                    double[] avg_dw = new double[inputSize];
                    for (int j = 0; j < inputSize; j++)
                        avg_dw[j] = (localGrads[j] + workerGrads[j]) / 2.0;
                    double avg_db = (localGrads[inputSize] + workerGrads[inputSize]) / 2.0;

                    // Update Weights (Initial w=0.5, b=0.5)
                    double[] w = new double[inputSize];
                    for (int j = 0; j < inputSize; j++)
                        w[j] = 0.5 - (0.01 * avg_dw[j]);
                    double b = 0.5 - (0.01 * avg_db);

                    // Save Model
                    int modelId = trainedModels.incrementAndGet();
                    Model newModel = new Model(w, b);
                    modelStore.put(modelId, newModel);
                    modelNames.put(modelId, "Model " + modelId); // Default Name

                    // RAFT Commit
                    synchronized (workerLock) {
                        workerOut.writeByte(0x07); // COMMIT_ENTRY
                        // Payload: [ModelID][InputSize][w...][b]
                        workerOut.writeInt(4 + 4 + (inputSize * 8) + 8);
                        workerOut.writeInt(modelId);
                        workerOut.writeInt(inputSize);
                        for (double val : w)
                            workerOut.writeDouble(val);
                        workerOut.writeDouble(b);
                        workerOut.flush();
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
                            double prediction = m.b;
                            for (int i = 0; i < inputSize; i++)
                                prediction += m.w[i] * inputVal[i];
                            out.writeUTF("Prediction: " + prediction);
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
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    static class TrainingCore {
        public static double[] computeGradients(double[][] inputs, double[] targets, int inputSize) {
            double[] w = new double[inputSize];
            Arrays.fill(w, 0.5);
            double b = 0.5;

            double[] grad_w = new double[inputSize];
            double grad_b = 0.0;

            for (int i = 0; i < inputs.length; i++) {
                double pred = b;
                for (int j = 0; j < inputSize; j++)
                    pred += w[j] * inputs[i][j];

                double error = targets[i] - pred;

                for (int j = 0; j < inputSize; j++)
                    grad_w[j] += -2 * inputs[i][j] * error;
                grad_b += -2 * error;
            }

            if (inputs.length > 0) {
                for (int j = 0; j < inputSize; j++)
                    grad_w[j] /= inputs.length;
                grad_b /= inputs.length;
            }

            // Return [dw1, dw2..., db]
            double[] result = Arrays.copyOf(grad_w, inputSize + 1);
            result[inputSize] = grad_b;
            return result;
        }
    }
}
