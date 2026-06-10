import numpy as np


def cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.0)

    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def cross_entropy_derivative(y_true, y_pred):
    batch_size = y_true.shape[0]

    return (y_pred - y_true) / batch_size

if __name__ == "__main__":
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])

    y_pred = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1]
    ])

    print(cross_entropy(y_true, y_pred))
    print(cross_entropy_derivative(y_true, y_pred))