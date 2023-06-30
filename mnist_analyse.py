import numpy as np
import pandas as pd

from pathlib import Path
import time

from neural_network.layers import DenseLayer
from neural_network.activation import Tanh
from neural_network.loss import MSE
from neural_network.utils.nparray import to_categorical
from neural_network.network import NeuralNetwork

# 1Epoch +/- 25s Training
EPOCHS = 12
LEARNING_RATE = 0.01

DATA_FOLDER = Path("mnist/")
TRAINING_DATA_PATH = DATA_FOLDER / Path("mnist_train.csv")
TEST_DATA_PATH = DATA_FOLDER / Path("mnist_test.csv")


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path, sep=",")

    X = np.array(data.drop(columns=["label"]).values)
    X = X.reshape((len(X), 1, 784))
    X = X / 255

    Y = np.array(data["label"].values)
    Y = Y.reshape((len(Y), 1, 1))
    Y = to_categorical(Y)

    return X, Y


def test(network: NeuralNetwork, path: Path) -> float:
    X, Y = load_csv(path)
    true_preds = 0

    for x, y in zip(X, Y):
        x = network.forward(x)
        true_preds += int(np.argmax(x) == np.argmax(y))

    return true_preds / Y.shape[0]


def main():

    # generate network
    network = NeuralNetwork(MSE())
    network.add_layer(DenseLayer(784, 100))
    network.add_layer(Tanh())
    network.add_layer(DenseLayer(100, 50))
    network.add_layer(Tanh())
    network.add_layer(DenseLayer(50, 10))
    network.add_layer(Tanh())
    network.random_init()

    X_train, Y_train = load_csv(TRAINING_DATA_PATH)

    for e in range(0, 13):
        # train network
        for lr in [0.1, 0.01]:
            network.random_init()
            start_training_time = time.time()
            network.train(X_train, Y_train, e, lr)
            end_training_time = time.time()

            training_time = end_training_time - start_training_time

            print(f"{lr} LEARNING_RATE")
            print(f"{e} training epochs")
            print(f"Training time: {training_time/60} min")
            print(f"{test(network, TEST_DATA_PATH)*100}% true predictions")
            print(end="\n" * 3)


if __name__ == "__main__":
    main()
