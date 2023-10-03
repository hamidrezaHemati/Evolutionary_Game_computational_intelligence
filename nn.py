import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):
        input_layer = layer_sizes[0]
        hidden_layer = layer_sizes[1]
        output_layer = layer_sizes[2]
        self.w0 = np.random.normal(size=(hidden_layer, input_layer))  # 20*5
        self.w1 = np.random.normal(size=(output_layer, hidden_layer))  # 1*20
        self.b0 = np.random.uniform(0, 0, (hidden_layer, 1))  # 20*1
        self.b1 = np.random.uniform(0, 0, (output_layer, 1))  # 1*1
        pass

    def activation(self, x):
        e = 2.71
        return 1 / (1 + (pow(e, (-1 * x))))

    def forward(self, input_layer):
        # x example: np.array([[0.1], [0.2], [0.3]])
        neuron_A0 = input_layer
        neuron_A1 = self.activation((self.w0 @ neuron_A0) + self.b0)
        neuron_A2 = self.activation((self.w1 @ neuron_A1) + self.b1)
        return neuron_A2
