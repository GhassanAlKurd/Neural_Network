import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List

from sklearn.datasets import fetch_openml


# ---------------------- ACTIVATIONS ---------------------- #

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def leaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, z, z * 0.01)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax over columns (each column = one sample).
    """
    shifted = z - np.max(z, axis=0, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=0, keepdims=True)


def one_hot_encode(x: np.ndarray, num_labels: int) -> np.ndarray:
    """
    x: shape (m,), integer labels in [0, num_labels)
    returns: shape (m, num_labels)
    """
    return np.eye(num_labels)[x]


def derivative(function_name: str, z: np.ndarray) -> np.ndarray:
    """
    Derivative w.r.t. z (pre-activation).
    """
    if function_name == "sigmoid":
        sig = sigmoid(z)
        return sig * (1 - sig)
    if function_name == "tanh":
        t = tanh(z)
        return 1 - np.square(t)
    if function_name == "relu":
        return (z > 0).astype(float)
    if function_name == "leaky_relu":
        return np.where(z > 0, 1.0, 0.01)
    raise ValueError(f"Unknown activation function: {function_name}")


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
}


# ---------------------- NEURAL NET ---------------------- #

class NN:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        activation: str,
        num_labels: int,
        architecture: List[int],
    ):
        """
        X, X_test: shape (n_features, m)
        y, y_test: shape (num_labels, m) one-hot
        activation: one of ACTIVATIONS keys for hidden layers
        architecture: list of hidden layer sizes, e.g. [128, 32]
        """
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation function: {activation}")

        if y.shape[0] != num_labels:
            raise ValueError(
                f"Label matrix and num_labels mismatch: y.shape[0]={y.shape[0]}, num_labels={num_labels}"
            )

        if y.shape[1] != X.shape[1]:
            raise ValueError("Number of samples in X and y must match")

        if y_test.shape[0] != num_labels or y_test.shape[1] != X_test.shape[1]:
            raise ValueError("Test label matrix shape must match test data and num_labels")

        # Store full datasets
        self.X = X
        self.X_test = X_test
        self.y = y
        self.y_test = y_test

        self.activation_name = activation
        self.activation = ACTIVATIONS[activation]
        self.num_labels = num_labels
        self.dropout_rate = 0.0   # default: no dropout
        self.l2_lambda = 0.0

        # full architecture: input -> hidden layers -> output
        self.architecture = [self.X.shape[0], *architecture, num_labels]
        self.L = len(self.architecture)  # number of layers incl. input
        self.m_train = self.X.shape[1]   # number of training samples

        self.parameters: Dict[str, np.ndarray] = {}
        self.layers: Dict[str, np.ndarray] = {}   # caches for the last forward pass
        self.costs: List[float] = []
        self.accuracies = {"train": [], "test": []}

        self.output: np.ndarray | None = None

    def initialize_parameters(self) -> None:
        """
        Initialize weights and biases with He or Xavier depending on activation.
        """
        for layer in range(1, self.L):
            fan_out = self.architecture[layer]
            fan_in = self.architecture[layer - 1]

            if self.activation_name in {"relu", "leaky_relu"}:
                # He initialization
                scale = np.sqrt(2.0 / fan_in)
            else:
                # Xavier / Glorot (simple variant)
                scale = np.sqrt(1.0 / fan_in)

            self.parameters[f"w{layer}"] = np.random.randn(fan_out, fan_in) * scale
            self.parameters[f"b{layer}"] = np.zeros((fan_out, 1))

    def forward(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Forward pass on arbitrary batch X, y.
        Stores intermediate activations in self.layers and final output in self.output.
        Returns the cross-entropy cost on this batch.
        """
        m = X.shape[1]
        params = self.parameters
        self.layers = {"a0": X}

        for l in range(1, self.L - 1):
            z = np.dot(params[f"w{l}"], self.layers[f"a{l-1}"]) + params[f"b{l}"]
            a = self.activation(z)

            # Apply dropout ONLY during training, not during inference
            if self.dropout_rate > 0:
                dropout_mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
                a = a * dropout_mask
                a = a / (1.0 - self.dropout_rate)  # inverted dropout scaling
                self.layers[f"d{l}"] = dropout_mask

            self.layers[f"z{l}"] = z
            self.layers[f"a{l}"] = a

        # output layer
        final_layer = self.L - 1
        z_final = np.dot(params[f"w{final_layer}"], self.layers[f"a{final_layer-1}"]) + params[f"b{final_layer}"]
        a_final = softmax(z_final)
        self.layers[f"z{final_layer}"] = z_final
        self.layers[f"a{final_layer}"] = a_final
        self.output = a_final

        # cross-entropy loss
        eps = 1e-9
        data_cost = -np.sum(y * np.log(a_final + eps)) / m

        # L2 cost
        l2_cost = 0
        for layer in range(1, self.L):
            l2_cost += np.sum(np.square(self.parameters[f"w{layer}"]))

        l2_cost = (self.l2_lambda / (2 * m)) * l2_cost

        cost = data_cost + l2_cost

        return cost

    def backpropagate(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward pass using stored forward activations (from the last batch).
        Fills and returns gradients for all layers.
        """
        if self.output is None:
            raise RuntimeError("Must call forward() before backpropagate().")

        m = y.shape[1]
        derivatives: Dict[str, np.ndarray] = {}
        final_layer = self.L - 1

        # dZ for output layer (softmax + cross-entropy simplification)
        dZ = self.output - y  # shape (num_labels, m)

        # output layer gradients
        derivatives[f"dW{final_layer}"] = np.dot(dZ, self.layers[f"a{final_layer-1}"].T) / m
        derivatives[f"db{final_layer}"] = np.sum(dZ, axis=1, keepdims=True) / m

        # propagate backwards
        dA_prev = np.dot(self.parameters[f"w{final_layer}"].T, dZ)

        for l in range(self.L - 2, 0, -1):
            z_l = self.layers[f"z{l}"]
            dZ = dA_prev * derivative(self.activation_name, z_l)

            # Apply dropout mask during backward pass
            if self.dropout_rate > 0:
                dZ = dZ * self.layers[f"d{l}"]
                dZ = dZ / (1.0 - self.dropout_rate)
                
            # L2 regularization term
            l2_term = (self.l2_lambda / m) * self.parameters[f"w{l}"]

            derivatives[f"dW{l}"] = np.dot(dZ, self.layers[f"a{l-1}"].T) / m + l2_term




            derivatives[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m

            if l > 1:
                dA_prev = np.dot(self.parameters[f"w{l}"].T, dZ)

        return derivatives

    def fit(self, lr: float = 0.01, epochs: int = 50, batch_size: int = 64, eval_every: int = 1) -> None:
       
        
        """
        Mini-batch gradient descent.
        batch_size: number of samples per batch
        eval_every: how often to recompute and print metrics (in epochs).
        """
        self.initialize_parameters()

        n_samples = self.m_train
        last_test_acc = None

        for epoch in tqdm(range(epochs), colour="BLUE", desc="Training"):
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            X_shuffled = self.X[:, indices]
            y_shuffled = self.y[:, indices]

            epoch_cost = 0.0

            # Iterate over mini-batches
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]
                m_batch = X_batch.shape[1]

                # forward + loss on batch
                cost_batch = self.forward(X_batch, y_batch)

                # backprop on batch
                gradients = self.backpropagate(y_batch)

                # parameter update
                for layer in range(1, self.L):
                    self.parameters[f"w{layer}"] -= lr * gradients[f"dW{layer}"]
                    self.parameters[f"b{layer}"] -= lr * gradients[f"db{layer}"]

                # accumulate cost (weighted by batch size)
                epoch_cost += cost_batch * (m_batch / n_samples)

            self.costs.append(epoch_cost)

            # metrics on full train/test sets
            if (epoch % eval_every == 0) or (epoch == epochs - 1) or (last_test_acc is None):
                train_accuracy = self.accuracy(self.X, self.y)
                test_accuracy = self.accuracy(self.X_test, self.y_test)
                last_test_acc = test_accuracy

                self.accuracies["train"].append(train_accuracy)
                self.accuracies["test"].append(test_accuracy)

                print(
                    f"Epoch: {epoch:3d} | Cost: {epoch_cost:.4f} | "
                    f"Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%"
                )
            else:
                # keep list lengths consistent, reuse last metrics
                self.accuracies["train"].append(self.accuracies["train"][-1])
                self.accuracies["test"].append(last_test_acc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Run a forward pass and return the class probability distribution
        for every column in X.
        """
        params = self.parameters
        a = X
        n_layers = self.L - 1

        for l in range(1, n_layers):
            z = np.dot(params[f"w{l}"], a) + params[f"b{l}"]
            a = self.activation(z)

        z = np.dot(params[f"w{n_layers}"], a) + params[f"b{n_layers}"]
        return softmax(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the most probable digit for each sample in X.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=0)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy (%) on given dataset.
        y: one-hot, shape (num_labels, m)
        """
        preds = self.predict(X)
        labels = np.argmax(y, axis=0)
        return float(np.mean(preds == labels) * 100.0)
    
    def save(self, filename: str) -> None:
        """
        Save all parameters (weights and biases) to a .npz file.
        """
        np.savez(filename, **self.parameters)
        print(f"Model parameters saved to {filename}")


    def load(self, filename: str) -> None:
        """
        Load parameters (weights and biases) from a .npz file.
        """
        loaded = np.load(filename)
        for key in loaded:
            self.parameters[key] = loaded[key]
        print(f"Model parameters loaded from {filename}")

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute confusion matrix for given dataset.
        Returns a 10x10 matrix for MNIST.
        """
        preds = self.predict(X)
        labels = np.argmax(y, axis=0)

        cm = np.zeros((self.num_labels, self.num_labels), dtype=int)
        for t, p in zip(labels, preds):
            cm[t, p] += 1

        return cm

    def show_misclassified(self, X: np.ndarray, y: np.ndarray, max_images: int = 9) -> None:
        """
        Show up to `max_images` misclassified digits.
        """
        preds = self.predict(X)
        labels = np.argmax(y, axis=0)

        mis_idx = np.where(preds != labels)[0]

        if len(mis_idx) == 0:
            print("No misclassified examples found.")
            return

        mis_idx = mis_idx[:max_images]

        plt.figure(figsize=(10, 10))
        side = int(np.sqrt(X.shape[0]))

        for i, idx in enumerate(mis_idx, 1):
            img = X[:, idx].reshape(side, side)
            plt.subplot(3, 3, i)
            plt.imshow(img, cmap="Greys")
            plt.title(f"True: {labels[idx]} | Pred: {preds[idx]}")
            plt.axis("off")

        plt.suptitle("Misclassified Examples", fontsize=16)
        plt.show()


    def show_prediction(self, X_single: np.ndarray) -> None:
        """
        Display one image and the model's probability distribution over 10 classes.
        X_single: shape (784,) or (784, 1)
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(-1, 1)

        # run forward pass manually
        probs = softmax(
            np.dot(self.parameters[f"w{self.L-1}"],
                   self.activation(
                       np.dot(self.parameters[f"w{self.L-2}"], X_single) +
                       self.parameters[f"b{self.L-2}"]
                   )
            ) + self.parameters[f"b{self.L-1}"]
        )

        pred = np.argmax(probs)

        # image
        side = int(np.sqrt(X_single.shape[0]))
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(X_single.reshape(side, side), cmap="Greys")
        plt.title(f"Predicted: {pred}")
        plt.axis("off")

        # probabilities
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(10), probs.flatten())
        plt.xticks(np.arange(10))
        plt.title("Class Probabilities")
        plt.tight_layout()
        plt.show()


    

    def plot_cost(self) -> None:
        plt.figure(figsize=(8, 4))
        plt.plot(self.costs)
        plt.title("Training Cost")
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.grid(True)
        plt.tight_layout()
        plt.show()





# ---------------------- MAIN / MNIST DEMO ---------------------- #

def main() -> None:
    # For reproducibility (optional)
    np.random.seed(42)

    # --------------------------
    # LOAD MNIST DATA
    # --------------------------
    mnist = fetch_openml(name="mnist_784", version=1, as_frame=False)
    data = mnist["data"].astype(np.float32)        # shape (70000, 784)
    labels = mnist["target"].astype(int)           # shape (70000,)

    # Show a random image
    random_index = np.random.randint(0, data.shape[0])
    img = data[random_index]
    side = int(np.sqrt(img.size))
    plt.imshow(img.reshape(side, side), cmap="Greys")
    plt.title(f"Sample label: {labels[random_index]}")
    plt.axis("off")
    plt.show()

    # --------------------------
    # TRAIN / TEST SPLIT
    # --------------------------
    split = 60000    # first 60k → train
    X_train = (data[:split] / 255.0).T              # (784, 60000)
    X_test = (data[split:] / 255.0).T               # (784, 10000)
    y_train = one_hot_encode(labels[:split], 10).T  # (10, 60000)
    y_test = one_hot_encode(labels[split:], 10).T   # (10, 10000)

    # --------------------------
    # NETWORK CONFIG
    # --------------------------
    hidden_layers = [128, 32]
    activation = "relu"
    epochs = 20
    lr = 0.003
    batch_size = 64

    # --------------------------
    # INITIALIZE MODEL
    # --------------------------
    nn = NN(
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        activation=activation,
        num_labels=10,
        architecture=hidden_layers,
    )

    # Enable dropout + L2 regularization
    nn.dropout_rate = 0.2        # 20% dropout on hidden layers
    nn.l2_lambda = 0.0005        # small L2 weight decay (standard)

    # --------------------------
    # TRAIN MODEL
    # --------------------------
    nn.fit(
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        eval_every=1
    )

    # --------------------------
    # SAVE TRAINED WEIGHTS
    # --------------------------
    nn.save("mnist_weights.npz")

    # --------------------------
    # PLOT COST
    # --------------------------
    nn.plot_cost()




    cm = nn.confusion_matrix(X_test, y_test)
    print("Confusion Matrix:")
    print(cm)

    # 2. Misclassified Examples
    nn.show_misclassified(X_test, y_test, max_images=9)

    # 3. Prediction Probabilities for a Single Image
    sample = X_test[:, 123]     # pick any index you want
    nn.show_prediction(sample)





if __name__ == "__main__":
    main()
