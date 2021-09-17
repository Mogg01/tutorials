import matplotlib.pyplot as plt
from ml_containers.neuron_types import PathNeuron, InputNeuron
from ml_containers.entities import NeuralCluster

from tensorflow.keras import losses
from tensorflow.keras.datasets import fashion_mnist, cifar100
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

NeuralCluster = NeuralCluster()

entity = InputNeuron(x_train[0])
entity.compile(optimizer='adam', loss=losses.MeanSquaredError())

entity.fit(x_train, x_train,
           epochs=100,
           shuffle=True,
           validation_data=(x_test, x_test))


encoded_imgs = entity._encoder(x_test[:10]).numpy()
decoded_imgs = entity._decoder(encoded_imgs[:10]).numpy()
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
