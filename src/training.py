#!/usr/bin/env python
# coding: utf-8


# In[ ]:
# Import Python packages
import os


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


# In[ ]:
# Constants
EPOCHS = 20
IMG_HEIGHT = 150
IMG_WIDHT = 150
BATCH_SIZE = 32
INPUT_SHAPE = (150, 150, 3)
VIEW_DATASET_IMAGES = False # set to True to view images

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


# In[ ]:
# Get data
if __name__ == "__main__":
    train_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET,
            validation_split = 0.2,
            image_size=(IMG_HEIGHT, IMG_WIDHT),
            batch_size=BATCH_SIZE,
            subset="training",
            seed=50,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET,
            validation_split=0.2,
            image_size=(IMG_HEIGHT, IMG_WIDHT),
            batch_size=BATCH_SIZE,
            subset="validation",
            seed=50,
    )

    # Verify data
    print("-" * 80)
    print("[INFO] : BatchDataset type objects returned.")
    print("[INFO] : Class names:", train_ds.class_names)

    # In[ ]:
    # View image data
    if VIEW_DATASET_IMAGES:
        print("-" * 80)
        print('[INFO] : Showing image from training dataset with 20% validation data.')

        for (image, label) in train_ds.take(1):
            # Plot the first 9 images
            plt.figure(figsize=(10, 10))
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(image[i].numpy().astype("uint8"))
                plt.title(train_ds.class_names[label[i]])
                plt.axis("Off")
            plt.show()
    else:
        pass


    # In[ ]:
    # Use caching and prefetching to optimize loading speed
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("-" * 80)
    print('[INFO] : Autotune setup complete.')


    # In[ ]:
    # Using pre-trained Resnet-50 layers model to train on our fire-dataset
    # here we are setting include_top as False, as we will add our own dense layers after resnet 50 last layer
    pre_trained_resnet_50 = tf.keras.applications.ResNet50(include_top = False,
                                                        input_shape = INPUT_SHAPE,
                                                        pooling = 'avg',
                                                        classes = 100,
                                                        weights = 'imagenet')

    # Here we want last 10 layers to be trainable so freezing first 40 layers
    x = 0
    for layer in pre_trained_resnet_50.layers:
        layer.trainable = False
        x+=1
        if x == 39:
            break

    print("-" * 80)
    print("[INFO] : Completed importing pre trained RESNET50 model weights.")


    # In[ ]:


    # Adding extra Dense layers after Resnet 50 last layer, we do this to increase
    # our models capability to categorise image as having fire or not having fire
    model = Sequential()
    model.add(pre_trained_resnet_50)
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4000, activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.summary()


    # In[ ]:


    # Using tensorflow's learning-rate-scheduler to change learning rate at each epoch
    # this will help us to find the best learning rate for our model
    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 1e-8 * 10**(epoch/20)
    )
    print("-" * 80)
    print("[INFO] : Learning rate callback scheduled.")

    # Using tensorflow's ModelCheckpoint to save best model having less validation loss
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = CHECKPOINT_FILEPATH, monitor = 'val_loss',
        save_best_only = True,
    )
    print("-" * 80)
    print("[INFO] : Model checkpoint callback scheduled.")

    # Using Adam optimizer to optimize our model to better learn on our dataset
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = 'binary_crossentropy',
        metrics = 'accuracy'
    )

    print("-" * 80)
    print("[INFO] : Model compiled using Adam optimizer.")


    # In[ ]:


    # Train model
    print("-" * 80)
    print("[INFO] : Starting model fitting with callbacks.")
    # Now time to train our model on fire dataset
    model_hist = model.fit(train_ds, validation_data = test_ds,
            epochs = EPOCHS,
            callbacks = [learning_rate_callback,  model_checkpoint_callback,],
    )

    # Save model weights
    try:
        print("[INFO] : Trying to save a .keras format model.")
        model.save(SAVE_KERAS_MODEL)
        print("[INFO] : Saved .keras model successfully.")
    except Exception as ex:
        print(ex)
        print("[INFO] : Trying to save a HDF5 model format.")
        model.save(SAVE_H5_MODEL,)
        print("[INFO] : Saved .h5 model successfully.")

    # # Model evaluation

    # In[ ]:


    # Evaluate model performance on test dataset
    model_hist.evaluate(test_ds)