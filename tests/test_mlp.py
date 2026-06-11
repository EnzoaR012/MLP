import unittest

import numpy as np

from mlp.activations import relu, relu_derivative, softmax
from mlp.losses import cross_entropy
from mlp.network import MLP
from mlp.optimizers import SGD


class ActivationTests(unittest.TestCase):
    def test_relu_and_derivative(self):
        values = np.array([-2.0, 0.0, 3.0])

        np.testing.assert_array_equal(relu(values), [0.0, 0.0, 3.0])
        np.testing.assert_array_equal(relu_derivative(values), [0.0, 0.0, 1.0])

    def test_softmax_rows_sum_to_one(self):
        probabilities = softmax(np.array([[2.0, 1.0], [0.0, 3.0]]))

        np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])


class NetworkTests(unittest.TestCase):
    def test_gradient_matches_numerical_approximation(self):
        inputs = np.array([[0.2, -0.1], [0.5, 0.3]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        network = MLP([2, 3, 2], seed=42)

        network.forward(inputs)
        analytical = network.backward(targets)["W1"][0, 0]

        original = network.parameters["W1"][0, 0]
        epsilon = 1e-5
        network.parameters["W1"][0, 0] = original + epsilon
        loss_plus = cross_entropy(targets, network.forward(inputs))
        network.parameters["W1"][0, 0] = original - epsilon
        loss_minus = cross_entropy(targets, network.forward(inputs))
        network.parameters["W1"][0, 0] = original
        numerical = (loss_plus - loss_minus) / (2 * epsilon)

        self.assertAlmostEqual(analytical, numerical, places=7)

    def test_network_learns_xor(self):
        inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        network = MLP([2, 8, 8, 2], seed=42)

        history = network.fit(
            inputs,
            targets,
            epochs=1500,
            batch_size=4,
            optimizer=SGD(learning_rate=0.1),
            seed=42,
            verbose=False,
        )

        self.assertLess(history["loss"][-1], history["loss"][0])
        self.assertEqual(network.evaluate(inputs, targets)["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
