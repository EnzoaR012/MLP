import numpy as np

from mlp.activations import relu, relu_derivative, softmax
from mlp.losses import cross_entropy, cross_entropy_derivative
from mlp.optimizers import SGD


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

    def fit(
        self,
        inputs,
        y_true,
        epochs=10,
        batch_size=32,
        optimizer=None,
        shuffle=True,
        seed=None,
        verbose=True,
        validation_data=None,
    ):
        """Treina a rede usando mini-batches."""
        if inputs.shape[0] != y_true.shape[0]:
            raise ValueError("inputs e y_true devem possuir o mesmo numero de amostras")

        if y_true.ndim != 2 or y_true.shape[1] != self.layer_sizes[-1]:
            raise ValueError(
                f"y_true deve possuir formato (amostras, {self.layer_sizes[-1]})"
            )

        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs e batch_size devem ser maiores que zero")

        optimizer = optimizer or SGD()
        random_generator = np.random.default_rng(seed)
        history = {"loss": [], "accuracy": []}
        if validation_data is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []

        sample_count = inputs.shape[0]

        for epoch in range(epochs):
            indices = np.arange(sample_count)
            if shuffle:
                random_generator.shuffle(indices)

            for start in range(0, sample_count, batch_size):
                batch_indices = indices[start : start + batch_size]
                batch_inputs = inputs[batch_indices]
                batch_targets = y_true[batch_indices]

                self.forward(batch_inputs)
                gradients = self.backward(batch_targets)
                optimizer.update(self.parameters, gradients)

            metrics = self.evaluate(inputs, y_true, batch_size=1024)
            history["loss"].append(metrics["loss"])
            history["accuracy"].append(metrics["accuracy"])

            validation_message = ""
            if validation_data is not None:
                validation_inputs, validation_targets = validation_data
                validation_metrics = self.evaluate(
                    validation_inputs, validation_targets, batch_size=1024
                )
                history["val_loss"].append(validation_metrics["loss"])
                history["val_accuracy"].append(validation_metrics["accuracy"])
                validation_message = (
                    f" - val_loss: {validation_metrics['loss']:.4f}"
                    f" - val_accuracy: {validation_metrics['accuracy']:.4f}"
                )

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"loss: {metrics['loss']:.4f} - "
                    f"accuracy: {metrics['accuracy']:.4f}"
                    f"{validation_message}"
                )

        return history

    def predict_proba(self, inputs):
        """Retorna as probabilidades previstas para cada classe."""
        return self.forward(inputs)

    def predict(self, inputs):
        """Retorna o indice da classe prevista para cada amostra."""
        return np.argmax(self.predict_proba(inputs), axis=1)

    def evaluate(self, inputs, y_true, batch_size=1024):
        """Calcula loss e acuracia para um conjunto de dados."""
        if inputs.shape[0] != y_true.shape[0]:
            raise ValueError("inputs e y_true devem possuir o mesmo numero de amostras")

        total_loss = 0.0
        total_correct = 0
        sample_count = inputs.shape[0]

        for start in range(0, sample_count, batch_size):
            end = start + batch_size
            batch_inputs = inputs[start:end]
            batch_targets = y_true[start:end]
            probabilities = self.predict_proba(batch_inputs)

            total_loss += cross_entropy(batch_targets, probabilities) * len(batch_inputs)
            total_correct += np.sum(
                np.argmax(probabilities, axis=1)
                == np.argmax(batch_targets, axis=1)
            )

        return {
            "loss": total_loss / sample_count,
            "accuracy": total_correct / sample_count,
        }
