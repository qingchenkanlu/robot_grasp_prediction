import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.backend import backend as K

base_dir = '/media/baxter/DataDisk/Cornell Grasps Dataset/original'
test_dir = os.path.join(base_dir, 'test')

BASE_FILENAME = 'pcd'
IMAGE_FILENAME_SUFFIX = 'png'
BBOX_FILENAME_SUFFIX = 'txt'
TARGET_IMAGE_WIDTH = 224
START_INSTANCE = 1000
INSTANCE_RANGE = 1035


def open_img(instance_num, target_size, base_filename, filename_suffix):
    img_filename = os.path.join(test_dir, base_filename + str(instance_num) + "r" + "." + filename_suffix)
    img = Image.open(img_filename)
    img = img.resize((target_size, target_size))

    return img


def img_to_array(img):
    # Converts a given RGB image (widht, height, 3) to a 1D array
    img_array = np.asarray(img, dtype='float32') / 255
    img_array = np.reshape(img_array, (-1))

    return img_array


def img_from_array(img_array, target_img_width):
    # Converts a given 1D array to the image size. Make sure to use the same image
    #  width to avoid overlapping or missing of pixel values
    img_array = np.reshape(img_array, (target_img_width, target_img_width, 3))
    img_from_array = Image.fromarray(img_array.astype('uint8'), 'RGB')

    return img_from_array


def open_bboxes(instance_num, base_filename, filename_suffix):
    filename = os.path.join(test_dir, base_filename + str(instance_num) + "cpos" + "." + filename_suffix)
    with open(filename) as f:
        bboxes = list(map(
            lambda coordinate: float(coordinate), f.read().strip().split()))

    return bboxes


def bboxes_to_grasps(box):
    x = (box[0] + (box[4] - box[0]) / 2) * 0.35
    y = (box[1] + (box[5] - box[1]) / 2) * 0.47
    tan = box[3] - box[1]
    h = box[3] + box[1]
    w = box[7] - box[6]
    grasp = [x, y, tan, h, w]

    return grasp


def load_data(start_instance, instance_range):
    x_train = []
    y_train = []

    for instance_num in range(start_instance, instance_range):
        bboxes = open_bboxes(instance_num, BASE_FILENAME, BBOX_FILENAME_SUFFIX)
        #print(bboxes)
        for box_num in range(0, len(bboxes), 8):
            y_train_temp = bboxes_to_grasps(bboxes[box_num:box_num+8])
            y_train.append(y_train_temp)

            img = open_img(instance_num, TARGET_IMAGE_WIDTH, BASE_FILENAME, IMAGE_FILENAME_SUFFIX)
            img_array = img_to_array(img)
            #print(img_array.shape)
            x_train.append(img_array)

    return x_train, y_train


def save_data(data_var_name, data_filename):
    print('Saving the data sets... :', data_filename)
    with open(data_filename, 'w') as f:
        f.write('data = %s' % data_var_name)
    # from file import score as my_list -> importing the saved datasets


def save_data_local():
    x_test_data, y_test_data = load_data(START_INSTANCE, INSTANCE_RANGE)
    print('Length of X_DATA', len(x_test_data))
    print('Length of Y_DATA', len(y_test_data))
    print('Saving datasets...')

    save_data(x_test_data, 'x_test_data.py')
    save_data(y_test_data, 'y_test_data.py')
    print('Saving datasets: DONE!')


def read_test_data():
    x_test_data, y_test_data = load_data(START_INSTANCE, INSTANCE_RANGE)

    return x_test_data, y_test_data


if __name__ == '__main__':
    read_test_data()