"""Arquitetura, forward pass, backpropagation e treinamento do MLP."""

import numpy as np


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
