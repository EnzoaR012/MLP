import numpy as np

from mlp.activations import relu, relu_derivative, softmax
from mlp.losses import cross_entropy_derivative


class MLP:
    """Rede neural totalmente conectada com arquitetura configuravel."""

    def __init__(self, layer_sizes, seed=None):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes deve conter ao menos entrada e saida")

        if any(not isinstance(size, int) or size <= 0 for size in layer_sizes):
            raise ValueError("todas as camadas devem possuir tamanho inteiro positivo")

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.parameters = {}

        self._initialize_parameters(seed)

    def _initialize_parameters(self, seed):
        """Inicializa pesos com He e vieses com zero."""
        random_generator = np.random.default_rng(seed)

        for layer_index in range(1, self.num_layers + 1):
            input_size = self.layer_sizes[layer_index - 1]
            output_size = self.layer_sizes[layer_index]

            self.parameters[f"W{layer_index}"] = (
                random_generator.standard_normal((input_size, output_size))
                * np.sqrt(2.0 / input_size)
            )
            self.parameters[f"b{layer_index}"] = np.zeros((1, output_size))

    def forward(self, inputs):
        """Executa o forward pass e retorna as probabilidades previstas."""
        if inputs.ndim != 2 or inputs.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"inputs deve possuir formato (amostras, {self.layer_sizes[0]})"
            )

        cache = {"A0": inputs}
        activation = inputs

        for layer_index in range(1, self.num_layers):
            weights = self.parameters[f"W{layer_index}"]
            biases = self.parameters[f"b{layer_index}"]

            linear_output = activation @ weights + biases
            activation = relu(linear_output)

            cache[f"Z{layer_index}"] = linear_output
            cache[f"A{layer_index}"] = activation

        output_layer = self.num_layers
        logits = (
            activation @ self.parameters[f"W{output_layer}"]
            + self.parameters[f"b{output_layer}"]
        )
        probabilities = softmax(logits)

        cache[f"Z{output_layer}"] = logits
        cache[f"A{output_layer}"] = probabilities
        self.cache = cache

        return probabilities

    def backward(self, y_true):
        """Calcula os gradientes dos parametros usando backpropagation."""
        if not hasattr(self, "cache"):
            raise RuntimeError("execute forward antes de backward")

        predictions = self.cache[f"A{self.num_layers}"]
        if y_true.shape != predictions.shape:
            raise ValueError(f"y_true deve possuir formato {predictions.shape}")

        gradients = {}
        delta = cross_entropy_derivative(y_true, predictions)

        for layer_index in range(self.num_layers, 0, -1):
            previous_activation = self.cache[f"A{layer_index - 1}"]

            gradients[f"W{layer_index}"] = previous_activation.T @ delta
            gradients[f"b{layer_index}"] = np.sum(delta, axis=0, keepdims=True)

            if layer_index > 1:
                delta = (
                    delta @ self.parameters[f"W{layer_index}"].T
                    * relu_derivative(self.cache[f"Z{layer_index - 1}"])
                )

        return gradients
