import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA

def load_tiny(img_size=(64, 64), pca_components=''):
    DATA_DIR_TRAIN = ""
    DATA_DIR_VAL = ""
    VAL_ANNOTATIONS = ""

    train_class_folders = sorted(os.listdir(DATA_DIR_TRAIN))
    df_val = pd.read_csv(VAL_ANNOTATIONS, sep="\t", header=None, names=["filename", "wnid", "x1", "y1", "x2", "y2"])
    val_class_folders = sorted(df_val["wnid"].unique())

    all_class_folders = sorted(set(train_class_folders) | set(val_class_folders))
    class_to_index = {class_name: idx for idx, class_name in enumerate(all_class_folders)}

    def list_train_image_files(data_dir):
        file_paths, labels = [], []
        for class_folder in train_class_folders:
            images_path = os.path.join(data_dir, class_folder, "images")
            if os.path.exists(images_path):
                for file in os.listdir(images_path):
                    file_path = os.path.join(images_path, file)
                    if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        file_paths.append(file_path)
                        labels.append(class_to_index[class_folder])
        return file_paths, labels

    train_image_paths, train_labels = list_train_image_files(DATA_DIR_TRAIN)
    train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))

    def list_val_image_files(data_dir):
        return [os.path.join(data_dir, file) for file in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, file)) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    val_image_paths = list_val_image_files(DATA_DIR_VAL)
    val_image_ds = tf.data.Dataset.from_tensor_slices(val_image_paths)

    val_labels = {row["filename"]: class_to_index[row["wnid"]] for _, row in df_val.iterrows() if row["wnid"] in class_to_index}

    def process_val_image(file_path):
        filename = tf.strings.split(file_path, os.sep)[-1].numpy().decode("utf-8")
        label = val_labels.get(filename, -1)
        return file_path, label

    val_filenames_with_labels = val_image_ds.map(lambda x: tf.py_function(process_val_image, [x], [tf.string, tf.int32]))
    val_filenames_with_labels = val_filenames_with_labels.filter(lambda x, y: tf.not_equal(y, -1))

    def decode_img(img_path, label):
        img = tf.io.read_file(img_path)
        try:
            img = tf.image.decode_jpeg(img, channels=3)
        except tf.errors.InvalidArgumentError:
            return None, None
        img = tf.image.resize(img, img_size) / 255.0
        return img, label

    def dataset_to_numpy(dataset):
        images, labels = [], []
        for img, label in dataset:
            if img is not None and label is not None:
                images.append(img.numpy())
                labels.append(label.numpy())
        return np.array(images), np.array(labels)

    train_ds = train_ds.map(lambda x, y: tf.py_function(decode_img, [x, y], [tf.float32, tf.int32]))
    val_ds = val_filenames_with_labels.map(lambda x, y: tf.py_function(decode_img, [x, y], [tf.float32, tf.int32]))

    x_train, y_train = dataset_to_numpy(train_ds)
    x_test, y_test = dataset_to_numpy(val_ds)

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    pca = PCA(n_components=pca_components)
    x_train = pca.fit_transform(x_train_flat)
    x_test = pca.transform(x_test_flat)

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return (x_train, y_train), (x_test, y_test)
