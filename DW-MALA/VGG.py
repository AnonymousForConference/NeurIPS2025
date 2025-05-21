import numpy as np
import tensorflow as tf
from sklearn.neighbors import KernelDensity
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import pairwise_distances
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def build_vgg_model():
    model = models.Sequential()
    
    def ConvBNReLU(nInputPlane, nOutputPlane, dropout_rate=0.0):
        model.add(layers.Conv2D(nOutputPlane, (3, 3), padding='same', activation=None, kernel_initializer=HeNormal()))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    MaxPooling = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

    ConvBNReLU(1, 64, dropout_rate=0.3)
    ConvBNReLU(64, 64)
    model.add(MaxPooling)

    ConvBNReLU(64, 128, dropout_rate=0.4)
    ConvBNReLU(128, 128)
    model.add(MaxPooling)

    ConvBNReLU(128, 256, dropout_rate=0.4)
    ConvBNReLU(256, 256, dropout_rate=0.4)
    ConvBNReLU(256, 256)
    model.add(MaxPooling)

    ConvBNReLU(256, 512, dropout_rate=0.4)
    ConvBNReLU(512, 512, dropout_rate=0.4)
    ConvBNReLU(512, 512)
    model.add(MaxPooling)

    ConvBNReLU(512, 512, dropout_rate=0.4)
    ConvBNReLU(512, 512, dropout_rate=0.4)
    ConvBNReLU(512, 512)
    model.add(MaxPooling)

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu', kernel_initializer=HeNormal()))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu', kernel_initializer=HeNormal()))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))

    optimizer = RMSprop(learning_rate=1e-3, decay=1e-6)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model