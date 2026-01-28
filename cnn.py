import numpy as np
from scipy.signal import correlate2d, convolve2d


def ReLU(x):
    return np.maximum(0, x)


def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def step(self, lr):
        pass


class Convolutional(Layer):
    def __init__(
        self, in_channels, out_channels, kernel_size, activation=None, seed=None
    ):
        rng = np.random.default_rng(seed)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation

        self.W = (
            rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size))
            * 0.1
        )
        self.b = np.zeros(out_channels)

        self.x_cache = None
        self.z_cache = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x_cache = x
        batch, _, h, w = x.shape
        k = self.kernel_size
        out_h, out_w = h - k + 1, w - k + 1

        z = np.zeros((batch, self.out_channels, out_h, out_w))

        for n in range(batch):
            for f in range(self.out_channels):
                for c in range(self.in_channels):
                    z[n, f] += correlate2d(x[n, c], self.W[f, c], mode="valid")
                z[n, f] += self.b[f]

        self.z_cache = z
        return self.activation(z) if self.activation else z

    def backward(self, grad_output):
        grad_z = (
            grad_output * (self.z_cache > 0) if self.activation is ReLU else grad_output
        )

        x = self.x_cache
        batch = x.shape[0]

        self.dW = np.zeros_like(self.W)
        for n in range(batch):
            for f in range(self.out_channels):
                for c in range(self.in_channels):
                    self.dW[f, c] += correlate2d(x[n, c], grad_z[n, f], mode="valid")

        self.db = np.sum(grad_z, axis=(0, 2, 3))

        grad_x = np.zeros_like(x)
        for n in range(batch):
            for f in range(self.out_channels):
                for c in range(self.in_channels):
                    grad_x[n, c] += convolve2d(grad_z[n, f], self.W[f, c], mode="full")

        return grad_x

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class MaxPool(Layer):
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.mask_cache = None
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        batch, channels, h, w = x.shape
        p = self.pool_size
        out_h, out_w = h // p, w // p

        x_reshaped = x[:, :, : out_h * p, : out_w * p].reshape(
            batch, channels, out_h, p, out_w, p
        )
        out = x_reshaped.max(axis=(3, 5))

        x_pool_view = x_reshaped.reshape(batch, channels, out_h, out_w, p * p)
        self.mask_cache = x_pool_view == x_pool_view.max(axis=4, keepdims=True)

        return out

    def backward(self, grad_output):
        batch, channels, h, w = self.input_shape
        p = self.pool_size
        out_h, out_w = h // p, w // p

        grad_expanded = np.broadcast_to(
            grad_output[:, :, :, :, np.newaxis], (batch, channels, out_h, out_w, p * p)
        )

        grad_masked = grad_expanded * self.mask_cache
        grad_x = grad_masked.reshape(batch, channels, out_h, p, out_w, p)
        grad_x = grad_x.transpose(0, 1, 2, 4, 3, 5).reshape(
            batch, channels, out_h * p, out_w * p
        )

        if h > out_h * p or w > out_w * p:
            grad_full = np.zeros(self.input_shape)
            grad_full[:, :, : out_h * p, : out_w * p] = grad_x
            return grad_full

        return grad_x


class Flatten(Layer):
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class Dense(Layer):
    def __init__(self, in_features, out_features, activation=None, seed=None):
        rng = np.random.default_rng(seed)
        self.activation = activation

        self.W = rng.standard_normal((in_features, out_features)) * 0.1
        self.b = np.zeros(out_features)

        self.x_cache = None
        self.z_cache = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x_cache = x
        z = x @ self.W + self.b
        self.z_cache = z
        return self.activation(z) if self.activation else z

    def backward(self, grad_output):
        grad_z = (
            grad_output * (self.z_cache > 0) if self.activation is ReLU else grad_output
        )
        self.dW = self.x_cache.T @ grad_z
        self.db = np.sum(grad_z, axis=0)
        return grad_z @ self.W.T

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class CNN:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_loss):
        for layer in reversed(self.layers):
            grad_loss = layer.backward(grad_loss)

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)

    def train(self, X, y, epochs=10, lr=0.01, batch_size=32):
        n_samples = y.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            epoch_loss = 0.0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                m = y_batch.shape[0]

                logits = self.forward(X_batch)
                probs = softmax(logits)

                epoch_loss += -np.log(probs[np.arange(m), y_batch] + 1e-10).mean()

                y_one_hot = np.zeros_like(probs)
                y_one_hot[np.arange(m), y_batch] = 1

                self.backward((probs - y_one_hot) / m)
                self.step(lr)

            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (n_samples // batch_size):.4f}"
            )

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    import time

    start = time.time()

    print("Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(int)
    X = X.reshape(-1, 1, 28, 28)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, y_train = X_train[:10000], y_train[:10000]
    X_test, y_test = X_test[:2500], y_test[:2500]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = CNN(
        [
            Convolutional(1, 8, 3, activation=ReLU, seed=42),
            MaxPool(2),
            Convolutional(8, 16, 3, activation=ReLU, seed=43),
            MaxPool(2),
            Flatten(),
            Dense(16 * 5 * 5, 10, seed=44),
        ]
    )

    model.train(X_train, y_train, epochs=30, lr=0.01, batch_size=32)
    end = time.time()
    print(f"Test accuracy: {model.accuracy(X_test, y_test) * 100:.2f}%")
    print(f"Execution time: {end - start:.2f} seconds")
