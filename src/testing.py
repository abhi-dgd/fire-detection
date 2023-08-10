#!/usr/bin/env python
# coding: utf-8
__doc__ = """Script to test the trained model. Path must be confirmed.
Images are provided randomly based on folder indexing and randomly generated
numbers."""

# In[]:
# Import Python packages
import os
import time


# Import installed libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('Tensorflow version', tf.version.VERSION)



__author__ = "Abhishek Degadwala"
__copyright__ = "None"
__credits__ = ["Abhishek Degadwala", "Galvin Fernandes", "Romanch Shah",]
__version__ = "1.0.0"
__maintainer__ = "Abhishek Degadwala"
__email__ = "degadwaa@mcmaster.ca"
__status__ = "Production"



# In[]:
# Constants
EPOCHS = 20
IMG_HEIGHT = 150
IMG_WIDHT = 150
BATCH_SIZE = 32
INPUT_SHAPE = (150, 150, 3)
VIEW_DATASET_IMAGES = False

# File paths
FIRE_DET_PATH = '/Users/abhidgd/Desktop/mcmaster/cps-dl/fire-detection'
DATASET = FIRE_DET_PATH + '/data/'
PRED_DATASET = FIRE_DET_PATH + '/predict/'
FIRE_IMG_PATH = DATASET + '/1'
NO_FIRE_IMG_PATH = DATASET + '/0'
FIRE_IMG = FIRE_IMG_PATH + '/1.jpg'
RAND_IMG = NO_FIRE_IMG_PATH + '/1.jpg'

# Output paths
CHECKPOINT_FILEPATH = FIRE_DET_PATH + '/temp/checkpoint'
KERAS_MODEL = FIRE_DET_PATH + '/model/retrained_imagenet.keras'
HDF5_MODEL = FIRE_DET_PATH + '/model/retrained_imagenet.h5'
SAVE_GRAPH = FIRE_DET_PATH + '/output/training_graph.png'


# In[ ]:
# Constants
EPOCHS = 20
IMG_HEIGHT = 150
IMG_WIDHT = 150
BATCH_SIZE = 32
INPUT_SHAPE = (150, 150, 3)
VIEW_DATASET_IMAGES = False

# File paths
FIRE_DET_PATH = '/Users/abhidgd/Desktop/mcmaster/cps-dl/fire-detection'
DATASET = FIRE_DET_PATH + '/data/'
FIRE_IMG_PATH = DATASET + '/1'
NO_FIRE_IMG_PATH = DATASET + '/0'
FIRE_IMG = FIRE_IMG_PATH + '/1.jpg'
RAND_IMG = NO_FIRE_IMG_PATH + '/1.jpg'

# Output paths
CHECKPOINT_FILEPATH = FIRE_DET_PATH + '/temp/checkpoint'
SAVE_KERAS_MODEL = FIRE_DET_PATH + '/model/retrained_imagenet.keras'
SAVE_H5_MODEL = FIRE_DET_PATH + '/model/retrained_imagenet.h5'


# In[ ]:
# Functions
def count_files_in_folders(parent_folder) -> dict:
    """Count files in given folder. Return dictionary with folder and count.
    
    Parameters
    ----------
    parent_folder: str;
        Folder who's sub folders and files are desired to be counted.
    
    Return
    ------
    file_count: dict;
        Python dict with folder name and number of files in folder. Within
        parent folder.
    """
    folder_names = os.listdir(parent_folder)
    file_count = {}

    for folder_name in folder_names:
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            file_count[folder_name] = len(
                [file for file in os.listdir(folder_path) if os.path.isfile(
                    os.path.join(folder_path, file)
                    )
                ]
            )
        else:
            file_count[folder_name] = 0

    return file_count


def preprocess_image(filename, img_shape = 150):
    """Funtion to read image and transform image to tensor.
    
    Parameters
    ----------
    filename: str;
        Image path to test model with.

    img_shape: int;
        Size of the image in pixels. Any given image will be resized to given
        dimension. Must match input image width and height. 
    """

    img = tf.io.read_file(filename) #read image
    img = tf.image.decode_image(img) # decode the image to a tensor
    img = tf.image.resize(img, size = [img_shape, img_shape]) # resize the image
    return img


def predict(model, filename, class_names: list):
    """Funtion to read image and give desired output with image.

    Parameters
    ----------
    model: keras object;
        Pretrained existing or custom trained model weights.
    
    filename: str;
        Image path to test model with.

    class_name: list;
        List of class names.
    """

    # Import the target image and preprocess it
    img = preprocess_image(filename)
    
    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    
    if len(pred[0]) > 1: # check for multi-class
        # if more than one output, take the max
        pred_class = class_names[pred.argmax()]
    else:
        # if only one output, round
        pred_class = class_names[int(tf.round(pred)[0][0])]

    # Plot the image and predicted class
    sh_image = plt.imread(filename)
    plt.imshow(sh_image)
    plt.title(f"Prediction: {pred_class}")
    plt.axis("Off")
    plt.show()


# In[]:
if __name__ == "__main__":
    # Make predictions using loaded model
    # Load custom retrained ResNet50 model
    loaded_model = tf.keras.models.load_model(KERAS_MODEL)
    easy_class_labels = ['Fire not found in image', 'Fire']
    fire_images: list = os.listdir(FIRE_IMG_PATH)
    no_fire_images: list = os.listdir(NO_FIRE_IMG_PATH)
    pred_imgs: list = os.listdir(PRED_DATASET)


    # In[]:
    # Generate random index number from range of list indices
    rand_test_image_idx = np.random.randint(0, len(pred_imgs), dtype=int)
    rand_test_image_path = PRED_DATASET + f'/{pred_imgs[rand_test_image_idx]}'

    print("-" * 80)
    print("[INFO] : Constants defined")
    print("[DEBUG] : Batch size = ", BATCH_SIZE)
    print("[DEBUG] : Epochs = ", EPOCHS)
    print("[DEBUG] : Input image height = ", IMG_HEIGHT)
    print("[DEBUG] : Input image width = ", IMG_WIDHT)
    print("[DEBUG] : Input image shape = ", INPUT_SHAPE)
    print(f"[DEBUG] : Saving model checkpoints to {CHECKPOINT_FILEPATH}",)
    print(f"[DEBUG] : Saving keras extension model to {SAVE_KERAS_MODEL}",)
    print(f"[DEBUG] : If required saving HDF5 extension model to {SAVE_H5_MODEL}",)
    print('-' * 80)
    print('[DEBUG] : Fires dataset size:\t', len(fire_images))
    print('-' * 80)
    print('[DEBUG] : Random images dataset size:\t', len(no_fire_images))
    print('-' * 80)
    print('[INFO] : Unseen real world image dataaset size:\t', len(pred_imgs))
    print('-' * 80)
    print('[DEBUG] : Randomly generated index number:\t', rand_test_image_idx)
    print('-' * 80)
    print(f'[INFO] : Running model on image \t ({pred_imgs[rand_test_image_idx]})')
    print(f'[INFO] : Image path\t ({rand_test_image_path})')
    print('-' * 80)
    time.sleep(3)

    # Generate predictions using randomly indexed image
    predict(loaded_model, rand_test_image_path, easy_class_labels)
