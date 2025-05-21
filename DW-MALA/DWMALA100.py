import matplotlib.pyplot as plt
import os
import time
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import pairwise_distances
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import random
from VGG import build_vgg_model
from dataloader.CIFAR100 import load_cifar100
from active_learning_loop import active_learning_loop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.stats import entropy
from keras import backend as K


print('cifar-100 start')



(x_train, y_train), (x_test, y_test) = load_cifar100()
model = build_vgg_model()
dwmala_train_acc, dwmala_test_acc = active_learning_loop(model, x_train, y_train, x_test, y_test)