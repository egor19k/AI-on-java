import java.util.Random;

public class N {
    public double[] I;
    public double[][] H;
    public double[] O;
    public double[][][] W;
    public double lr = 0.4;
    public String f = "w";
    public int layers;

    public static Random rand = new Random();

    public N(int i, int h, int layers, int o) {
        this.layers = layers;

        I = new double[i];
        O = new double[o];
        H = new double[layers][h];

        W = new double[layers + 1][][];
        for (int L = 0; L < layers + 1; L++) {
            int rows = (L == 0) ? i : h;
            int cols = (L == layers) ? o : h;
            W[L] = new double[rows][cols];
            for (int a = 0; a < rows; a++) {
                for (int b = 0; b < cols; b++) {
                    W[L][a][b] = rand.nextDouble() * 2 - 1;
                }
            }
        }
    }

    public double s(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double sd(double x) {
        return x * (1.0 - x);
    }

    public double[] forward(double[] X) {
        for (int i = 0; i < I.length; i++) {
            I[i] = (i < X.length) ? X[i] : 0;
        }

        double[] prev = I;
        for (int L = 0; L < W.length; L++) {
            double[] next_layer = (L == W.length - 1) ? O : H[L];
            for (int j = 0; j < next_layer.length; j++) {
                double sum = 0;
                for (int i = 0; i < prev.length; i++) {
                    sum = sum + prev[i] * W[L][i][j];
                }
                next_layer[j] = s(sum);
            }
            prev = next_layer;
        }

        return O.clone();
    }

    public double backprop(double[] T) {
        double[][] gradients = new double[W.length][];

        double[] OE = new double[O.length];
        for (int k = 0; k < O.length; k++) {
            OE[k] = (k < T.length ? T[k] : 0) - O[k];
        }

        double[] OD = new double[O.length];
        for (int k = 0; k < O.length; k++) {
            OD[k] = OE[k] * sd(O[k]);
        }
        gradients[W.length - 1] = OD;

        for (int L = W.length - 2; L >= 0; L--) {
            double[] current;
            if (L == W.length - 2) {
                current = O;
            } else {
                current = H[L + 1];
            }

            double[] prev;
            if (L == 0) {
                prev = I;
            } else {
                prev = H[L - 1];
            }

            double[] grad = new double[prev.length];

            for (int j = 0; j < prev.length; j++) {
                double sum = 0;
                for (int k = 0; k < current.length; k++) {
                    if (j < W[L + 1].length && k < W[L + 1][j].length && k < gradients[L + 1].length) {
                        sum = sum + gradients[L + 1][k] * W[L + 1][j][k];
                    }
                }
                grad[j] = sum * sd(prev[j]);
            }
            gradients[L] = grad;
        }

        for (int L = 0; L < W.length; L++) {
            double[] prev;
            if (L == 0) {
                prev = I;
            } else {
                prev = H[L - 1];
            }

            double[] next_layer;
            if (L == W.length - 1) {
                next_layer = O;
            } else {
                next_layer = H[L];
            }

            for (int i = 0; i < prev.length; i++) {
                for (int j = 0; j < next_layer.length; j++) {
                    if (i < W[L].length && j < W[L][i].length && j < gradients[L].length) {
                        W[L][i][j] = W[L][i][j] + lr * gradients[L][j] * prev[i];
                    }
                }
            }
        }

        double err = 0;
        for (int k = 0; k < O.length; k++) {
            err = err + OE[k] * OE[k];
        }
        return err / O.length;
    }

    public void train(double[][][] D, int E) {
        for (int e = 0; e < E; e++) {
            for (int s = 0; s < D.length; s++) {
                forward(D[s][0]);
                backprop(D[s][1]);
            }
        }
    }
    public void train(double[][][] D) {
        train(D, 1000);
    }

    public String save() {
        StringBuilder sb = new StringBuilder();
        sb.append(layers).append("|");
        sb.append(I.length).append("|");
        sb.append(H.length > 0 ? H[0].length : 0).append("|");
        sb.append(O.length).append("|");

        for (int L = 0; L < W.length; L++) {
            for (int i = 0; i < W[L].length; i++) {
                for (int j = 0; j < W[L][i].length; j++) {
                    sb.append(W[L][i][j]);
                    if (!(L == W.length - 1 && i == W[L].length - 1 && j == W[L][i].length - 1)) {
                        sb.append(",");
                    }
                }
            }
        }
        return sb.toString();
    }

    public void load(String data) {
        try {
            String[] parts = data.split("\\|");
            if (parts.length != 5) {
                throw new RuntimeException("Invalid data format");
            }

            int loadedLayers = Integer.parseInt(parts[0]);
            int inputSize = Integer.parseInt(parts[1]);
            int hiddenSize = Integer.parseInt(parts[2]);
            int outputSize = Integer.parseInt(parts[3]);

            this.layers = loadedLayers;
            this.I = new double[inputSize];
            this.O = new double[outputSize];
            this.H = new double[loadedLayers][hiddenSize];

            String[] weights = parts[4].split(",");
            int idx = 0;

            this.W = new double[loadedLayers + 1][][];
            for (int L = 0; L < loadedLayers + 1; L++) {
                int rows = (L == 0) ? inputSize : hiddenSize;
                int cols = (L == loadedLayers) ? outputSize : hiddenSize;
                W[L] = new double[rows][cols];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        if (idx < weights.length) {
                            W[L][i][j] = Double.parseDouble(weights[idx++]);
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error loading: " + e.getMessage());
        }
    }
}
