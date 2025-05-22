import matplotlib.pyplot as plt
import os
import time
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KernelDensity
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import pairwise_distances
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from scipy.spatial.distance import cdist
from dataloader.CIFAR10 import initialize_labeled_data

seed_value = ''
np.random.seed(seed_value)

def sample_from_gmm(num_samples, latent_dim='', num_modes=''):
    means = np.random.randn(num_modes, latent_dim) * 2.0
    cov = np.array([np.eye(latent_dim) for _ in range(num_modes)])
    samples = []
    for _ in range(num_samples):
        k = np.random.randint(0, num_modes)
        sample = np.random.multivariate_normal(means[k], cov[k])
        samples.append(sample)
    return np.array(samples)

def mala_sampling_gmm(gmm_samples, step_size=''):
    mala_samples = []
    for z in gmm_samples:
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        for _ in range(1):
            with tf.GradientTape() as tape:
                tape.watch(z)
                loss = tf.reduce_sum(tf.square(z))  # Dummy potential
            grad = tape.gradient(loss, z)
            noise = tf.random.normal(shape=z.shape)
            z = z + 0.5 * step_size * grad + step_size * noise
        mala_samples.append(z.numpy())
    return np.array(mala_samples)

def compute_density_score_latent(samples):
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(samples)
    log_density = kde.score_samples(samples)
    return np.exp(log_density)

def active_learning_loop(model, x_train, y_train, x_test, y_test, iterations=10, initial_size='', sample_size='', latent_dim='', select_size='', alpha=''):
    train_accuracies = []
    test_accuracies = []
    labeled_data, unlabeled_data = initialize_labeled_data(x_train, y_train, initial_size)
    for i in range(iterations):
        if len(unlabeled_data[0]) == 0:
            break
        if i == 0:
            model.fit(labeled_data[0], labeled_data[1], epochs='', batch_size=32, verbose=0)
        else:
            gmm_samples = sample_from_gmm(num_samples=len(unlabeled_data[0]), latent_dim=latent_dim)
            mala_samples = mala_sampling_gmm(gmm_samples, step_size='')
            labeled_data_latent = sample_from_gmm(len(labeled_data[0]), latent_dim=latent_dim)
            diversity_scores = []
            for sample in mala_samples:
                distances = cdist(sample.reshape(1, -1), labeled_data_latent.reshape(len(labeled_data_latent), -1), metric='euclidean')
                diversity_scores.append(distances.min())
            top_indices = np.argsort(diversity_scores)[-select_size:]
            top_samples = mala_samples[top_indices]
            density_scores = compute_density_score_latent(top_samples)
            v_scores = [(d ** alpha ) * (s ** (1 - alpha)) for d, s in zip(diversity_scores, density_scores)]
            selected_indices = np.argsort(v_scores)[-sample_size:]
            selected_data = unlabeled_data[0][selected_indices]
            selected_labels = unlabeled_data[1][selected_indices]
            labeled_data = (np.vstack((labeled_data[0], selected_data)), np.vstack((labeled_data[1], selected_labels)))
            unlabeled_data = (np.delete(unlabeled_data[0], selected_indices, axis=0), np.delete(unlabeled_data[1], selected_indices, axis=0))
            model.fit(labeled_data[0], labeled_data[1], epochs='', batch_size=32, verbose=0)
        train_accuracy = model.evaluate(labeled_data[0], labeled_data[1], verbose=0)[1]
        test_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(f"DW MALA GMM Iteration {i+1}/{iterations}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    return train_accuracies, test_accuracies