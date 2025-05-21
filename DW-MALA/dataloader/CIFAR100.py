import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.metrics import pairwise_distances
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import random
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.stats import entropy
from keras import backend as K

seed_value = 42
np.random.seed(seed_value)

def load_cifar100(pca_dim=''):
    
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)

    
    pca = PCA(n_components=pca_dim)
    x_train = pca.fit_transform(x_train_flat)
    x_test = pca.transform(x_test_flat)

    return (x_train, y_train), (x_test, y_test)

def initialize_labeled_data(x_pool, y_pool, initial_size=''):
    
    total_indices = np.arange(len(x_pool))
    labeled_indices = np.random.choice(total_indices, size=initial_size, replace=False)
    unlabeled_indices = np.setdiff1d(total_indices, labeled_indices)

    x_labeled = x_pool[labeled_indices]
    y_labeled = y_pool[labeled_indices]
    x_unlabeled = x_pool[unlabeled_indices]
    y_unlabeled = y_pool[unlabeled_indices]

    return (x_labeled, y_labeled), (x_unlabeled, y_unlabeled)