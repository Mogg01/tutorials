import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Reshape
from ml_containers.neuron_types import PathNeuron, InputNeuron


class NeuralCluster:
    def __init__(self):
        self._neurons = []
        for _ in range(100):
            self._add_neuron()

    def _add_neuron(self):
        self._neurons.append(PathNeuron())

    def call(self, x):
        pass


