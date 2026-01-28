import numpy as np


def ReLU(x):
    return np.maximum(0, x)


def softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)


def cross_entropy_from_logits(logits, y):
    probs = softmax(logits)
    return -np.log(probs[np.arange(y.shape[0]), y]).mean()


class Layer:
    def __init__(self, in_features, out_features, activation=None, seed=None):
        rng = np.random.default_rng(seed)
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        if activation is ReLU:
            scale = np.sqrt(2.0 / in_features)
        else:
            scale = np.sqrt(1.0 / in_features)

        self.W = rng.standard_normal((in_features, out_features)) * scale
        self.b = np.zeros((out_features,), dtype=float)

        self.x_cache = None
        self.z_cache = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x_cache = x
        z = x @ self.W + self.b
        self.z_cache = z
        if self.activation is None:
            return z
        return self.activation(z)

    def backward(self, grad_output):
        if self.activation is ReLU:
            grad_z = grad_output * (self.z_cache > 0).astype(float)
        else:
            grad_z = grad_output

        self.dW = self.x_cache.T @ grad_z
        self.db = np.sum(grad_z, axis=0)

        return grad_z @ self.W.T

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class NeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_loss):
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)

    def train(self, X, y, epochs=1000, lr=0.01):
        m = y.shape[0]

        for i in range(epochs):
            logits = self.forward(X)
            probs = softmax(logits)

            y_one_hot = np.zeros_like(probs)
            y_one_hot[np.arange(m), y] = 1

            grad_loss = (probs - y_one_hot) / m

            self.backward(grad_loss)
            self.step(lr)


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml

    print("Loading MNIST...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X, y = X / 255.0, y.astype(int)

    X_train, y_train = X[:1600], y[:1600]
    X_test, y_test = X[1600:2000], y[1600:2000]

    model = NeuralNet([Layer(784, 64, ReLU, seed=42), Layer(64, 10, seed=43)])

    print("Training...")
    model.train(X_train, y_train, epochs=1000, lr=0.1)

    print("Evaluating...")
    predictions = np.argmax(softmax(model.forward(X_test)), axis=1)
    print(f"Test accuracy: {np.mean(predictions == y_test) * 100:.2f}%")
