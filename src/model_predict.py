__doc__ = """Use trained model to make predictions on test data or real world
data.
"""

import os

import tensorflow as tf

import notebook


# specifying path to sample image from list of test images.
fire_img = notebook.FIRE_IMG
rand_img = notebook.RAND_IMG

# predefining class names so not to confuse with the output
class_names = ['Not-fire','Fire']

# Load model
loaded_model = tf.keras.models.load_model(notebook.SAVE_KERAS_MODEL)

# reading the input and checking the output
for images in os.listdir(notebook.FIRE_DET_PATH):
    print(len(images))
    print(images)
    notebook.pred_and_plot(loaded_model, fire_img, class_names)
