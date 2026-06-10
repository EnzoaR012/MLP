import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    shifted_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted_z)

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


if __name__ == "__main__":
    z = np.array([-2, -1, 0, 1, 2])

    print(relu(z))
    print(relu_derivative(z))

    logits = np.array([
        [2.0, 1.0, 0.1],
        [1.0, 3.0, 0.5]
    ])

    probabilities = softmax(logits)

    print(probabilities)
    print(np.sum(probabilities, axis=1))