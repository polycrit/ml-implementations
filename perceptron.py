import numpy as np


def train_perceptron_2d(X, y, n_iter=100):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(n_iter):
        changed = False
        for i in range(len(X)):
            if y[i] * (np.dot(w, X[i]) + b) <= 0:
                w += y[i] * X[i]
                b += y[i]
                changed = True

        if not changed:
            break

    def predict(x):
        x = np.asarray(x)
        scores = x @ w + b
        out = np.sign(scores)
        out[out == 0] = 1
        return out

    return predict


if __name__ == "__main__":
    from generate_dataset_2d import linear_separable, plot_decision_boundary
    import matplotlib.pyplot as plt

    ds = linear_separable(n=200, class_sep=3.0, seed=42)

    predict = train_perceptron_2d(ds.X, ds.y, n_iter=100)

    accuracy = np.mean(predict(ds.X) == ds.y)
    print(f"Training accuracy: {accuracy * 100:.2f}%")

    plot_decision_boundary(ds, predict, title="Perceptron")
    plt.show()
