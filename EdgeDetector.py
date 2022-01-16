import cv2
import numpy as np
from typing import List
from os.path import isfile
import tensorflow as tf


def create_dataset(
        input_images: List[str],
        kernel_size: int) -> (np.array, np.array):
    images_size = 0
    for input_image in input_images:
        image = cv2.imread(input_image)
        height, width, depth = image.shape
        images_size += width * height
    result_data: np.array = np.zeros((images_size, kernel_size, kernel_size))
    result_labels: np.array = np.zeros(images_size)
    current_kernel_index = 0
    for input_image in input_images:
        cv_image = cv2.imread(input_image)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edge_image = cv2.Canny(
            image=gray_image,
            threshold1=100,
            threshold2=200).astype(np.float32)
        edge_image /= 255
        height, width = gray_image.shape
        padding: int = kernel_size // 2
        gray_image_padding = np.pad(
            array=gray_image,
            pad_width=padding,
            mode='edge').astype(np.float32)
        gray_image_padding /= 255
        for image_y in range(height):
            for image_x in range(width):
                x_pos_begin = image_x
                x_pos_end = image_x + kernel_size
                y_pos_begin = image_y
                y_pos_end = image_y + kernel_size
                image_segment: np.array = gray_image_padding[y_pos_begin:y_pos_end, x_pos_begin:x_pos_end]
                result_data[current_kernel_index] = image_segment
                result_labels[current_kernel_index] = edge_image[image_y, image_x]
                current_kernel_index += 1
    return result_data, result_labels


def load_data(
        data_file_name: str,
        labels_file_name: str,
        file_list: List[str],
        kernel_size: int) -> (np.array, np.array):
    if not isfile(data_file_name) or not isfile(labels_file_name):
        data, labels = create_dataset(
            input_images=file_list,
            kernel_size=kernel_size)
        with open(data_file_name, 'wb') as f:
            np.save(f, data)
        with open(labels_file_name, 'wb') as f:
            np.save(f, labels)
    with open(data_file_name, 'rb') as f:
        data = np.load(f)
    with open(labels_file_name, 'rb') as f:
        labels = np.load(f)
    return data, labels


class EdgeDetector:
    def __train_model(self,
                      train_file_list: List[str],
                      test_file_list: List[str]):
        train_x_file_name = f'train_x_{self.kernel_size}x{self.kernel_size}.npy'
        train_y_file_name = f'train_y_{self.kernel_size}x{self.kernel_size}.npy'
        train_x, train_y = load_data(
            data_file_name=train_x_file_name,
            labels_file_name=train_y_file_name,
            file_list=train_file_list,
            kernel_size=self.kernel_size)
        test_x_file_name = f'test_x_{self.kernel_size}x{self.kernel_size}.npy'
        test_y_file_name = f'test_y_{self.kernel_size}x{self.kernel_size}.npy'
        test_x, test_y = load_data(
            data_file_name=test_x_file_name,
            labels_file_name=test_y_file_name,
            file_list=test_file_list,
            kernel_size=self.kernel_size)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.kernel_size, self.kernel_size)),
            tf.keras.layers.Dense((self.kernel_size ** 2) * 6, activation='softmax'),
            tf.keras.layers.Dense(1, activation='relu')
        ])
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['accuracy'])
        self.model.fit(train_x, train_y, epochs=3)
        model_file_name = f'model_{self.kernel_size}x{self.kernel_size}.h5'
        self.model.save(model_file_name)
        self.model.evaluate(test_x, test_y)

    def __init__(self,
                 train_file_list: List[str],
                 test_file_list: List[str],
                 kernel_size: int):
        self.kernel_size = kernel_size
        model_file_name = f'model_{self.kernel_size}x{self.kernel_size}.h5'
        if isfile(model_file_name):
            self.model = tf.keras.models.load_model(model_file_name)
        else:
            self.__train_model(
                train_file_list=train_file_list,
                test_file_list=test_file_list)

    def detect(self, input_image: np.array) -> np.array:
        height, width = input_image.shape
        padding: int = self.kernel_size // 2
        input_image_padding = np.pad(
            array=input_image,
            pad_width=padding,
            mode='edge').astype(np.float32)
        input_image_padding /= 255
        input_image_segments = np.zeros(
            (width * height, self.kernel_size, self.kernel_size),
            dtype=np.float32)
        current_kernel_index = 0
        for image_y in range(height):
            for image_x in range(width):
                x_pos_begin = image_x
                x_pos_end = image_x + self.kernel_size
                y_pos_begin = image_y
                y_pos_end = image_y + self.kernel_size
                image_segment: np.array = input_image_padding[y_pos_begin:y_pos_end, x_pos_begin:x_pos_end]
                input_image_segments[current_kernel_index] = image_segment
                current_kernel_index += 1
        predictions = self.model.predict(input_image_segments)
        return np.reshape(predictions, (height, width)) * 255
