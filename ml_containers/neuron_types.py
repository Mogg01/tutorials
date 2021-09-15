# from tensorflow.keras import Model, Sequential
# from tensorflow.keras.layers import Dense, Flatten, InputLayer, Reshape
import numpy as np
from keras import Model, Sequential
from keras.layers import Dense, Flatten, InputLayer, Reshape, Conv2D


class PathNeuron(Model):
    def __init__(self):
        super(PathNeuron, self).__init__()

        self._model = Sequential(
            [
                Dense(1024, activation='relu'),
                Dense(512, activation='relu'),
                Dense(256, activation='relu'),
                Dense(512, activation='relu'),
                Dense(1024, activation='relu')
            ]
        )

    def call(self, x):
        return self._model(x)


class InputNeuron(Model):
    def __init__(self, observation):
        super(InputNeuron, self).__init__()
        self._id = None
        self._memories = None
        self._path = []
        self._encoder = Sequential(
            [
                InputLayer(observation.shape),
                # Conv2D(16, (3, 3), 2),
                Conv2D(32, (3, 3), 2),
                Conv2D(64, (3, 3), 2),
                Conv2D(128, (3, 3), 2),
                Conv2D(256, (3, 3), 2),
                Flatten(),
                Dense(1024, activation='relu'),
                Dense(512, activation='relu'),
                Dense(256, activation='relu')

            ]
        )
        self._decoder = Sequential(
            [
                Dense(256, activation='sigmoid'),
                Dense(512, activation='sigmoid'),
                Dense(1024, activation='sigmoid'),
                Dense(np.prod(observation.shape), activation='sigmoid'),
                Reshape(observation.shape)
            ]
        )

    def _encode(self, x):
        return self._encoder(x)

    def _decode(self, x):
        return self._decoder(x)

    def call(self, x):
        encoded = self._encode(x)
        decoded = self._decode(encoded)
        return decoded
