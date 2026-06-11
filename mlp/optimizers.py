class SGD:
    """Atualiza os parametros usando stochastic gradient descent."""

    def __init__(self, learning_rate=0.01):
        if learning_rate <= 0:
            raise ValueError("learning_rate deve ser maior que zero")

        self.learning_rate = learning_rate

    def update(self, parameters, gradients):
        """Atualiza os parametros diretamente usando seus gradientes."""
        if parameters.keys() != gradients.keys():
            raise ValueError("parameters e gradients devem possuir as mesmas chaves")

        for name in parameters:
            parameters[name] -= self.learning_rate * gradients[name]
