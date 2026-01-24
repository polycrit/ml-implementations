import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(data, weights, bias):
    z = np.dot(weights.T, data) + bias
    return sigmoid(z)


def negative_log_likelihood_loss(g, a):
    positive_label = int(a == 1)

    term1 = positive_label * np.log(g)
    term2 = (1 - positive_label) * np.log(1 - g)

    return -(term1 + term2)


def logistic_regression_loss(data, labels, weights, bias, lam):
    s = 0
    n = data.shape[0]

    for i in range(n):
        pred = predict(data[i], weights, bias)
        nll = negative_log_likelihood_loss(pred, labels[i])
        s += nll + 0.5 * lam * np.dot(weights.T, weights)

    return 1 / n * s


def lr_loss_grad_weights(data, labels, weights, bias, lam):
    s = 0
    n = data.shape[0]

    for i in range(n):
        pred = predict(data[i], weights, bias)
        diff = pred - labels[i]

        s += diff * data[i]

    return 1 / n * s + lam * weights


def lr_loss_grad_bias(data, labels, weights, bias):
    s = 0
    n = data.shape[0]

    for i in range(n):
        pred = predict(data[i], weights, bias)
        diff = pred - labels[i]

        s += diff

    return 1 / n * s


def lr_gradient_descent(data, labels, learning_rate, lam, epsilon, max_iter):
    t = 0
    weights_prev = np.zeros(data.shape[1])
    bias_prev = 0
    loss_prev = logistic_regression_loss(data, labels, weights_prev, bias_prev, lam)
    while t < max_iter:
        t += 1
        weights_curr = weights_prev - learning_rate * lr_loss_grad_weights(
            data, labels, weights_prev, bias_prev, lam
        )
        bias_curr = bias_prev - learning_rate * lr_loss_grad_bias(
            data, labels, weights_prev, bias_prev
        )

        loss_curr = logistic_regression_loss(data, labels, weights_curr, bias_curr, lam)

        if np.abs(loss_curr - loss_prev) < epsilon:
            break

        weights_prev = weights_curr
        bias_prev = bias_curr
        loss_prev = loss_curr

    return weights_curr, bias_curr


def train_lr_2d(data, labels, learning_rate, lam, epsilon, max_iter):
    weights, bias = lr_gradient_descent(
        data, labels, learning_rate, lam, epsilon, max_iter
    )

    def predict(x):
        x = np.asarray(x)
        scores = x @ weights + bias
        return np.where(scores > 0, 1, -1)

    return predict
