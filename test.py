from sklearn.datasets import fetch_openml
from nn import NeuralNet, Layer, ReLU, softmax
import numpy as np

print("Loading MNIST data...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

y = y.astype(int)
X = X / 255.0

X_train = X[:1600]
y_train = y[:1600]

X_test = X[1600:2000]
y_test = y[1600:2000]

classifier = NeuralNet([Layer(784, 64, ReLU, seed=42), Layer(64, 10, None, seed=42)])

print("Training...")
classifier.train(X_train, y_train, epochs=1000, lr=0.1)

print("Evaluating...")
logits = classifier.forward(X_test)
probs = softmax(logits)
predictions = np.argmax(probs, axis=1)

accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.2%}")
