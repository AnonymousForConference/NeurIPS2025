from tensorflow.keras.datasets import cifar10
import random
import numpy as np

seed_value = 42
np.random.seed(seed_value)

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)


def initialize_labeled_data(x_train, y_train, initial_size=''):
    labeled_indices = random.sample(range(len(x_train)), initial_size)
    unlabeled_indices = [i for i in range(len(x_train)) if i not in labeled_indices]
    labeled_data = (x_train[labeled_indices], y_train[labeled_indices])
    unlabeled_data = (x_train[unlabeled_indices], y_train[unlabeled_indices])
    return labeled_data, unlabeled_data