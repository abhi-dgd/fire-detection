__doc__ = """"""

# Import Python packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Import installed libraries
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


import config
import auxiliary

if __name__ == "__main__":
    #%% Dataset breakdown
    file_info : dict = auxiliary.count_files_in_folders(config.DATASET)
    for folder, count in file_info.items():
        print(f'[INFO] : Found {count} images in {folder} subdir')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET, validation_split=0.2,
        image_size=(config.IMG_HEIGHT, config.IMG_WIDHT),
        batch_size=config.TRAIN_BATCH_SIZE, subset="training",
        seed=20,)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET, validation_split=0.2,
        image_size=(config.IMG_HEIGHT, config.IMG_WIDHT),
        batch_size=config.VAL_BATCH_SIZE, subset="validation",
        seed=20,)
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET, validation_split=0.2,
        image_size=(config.IMG_HEIGHT, config.IMG_WIDHT),
        batch_size=config.VAL_BATCH_SIZE, subset="validation",
        seed=20,)

    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Using pre-trained Resnet-50 layers model to train on our fire-dataset
    # here we are setting include_top as False, as we will add our own dense layers
    # after resnet 50 last layer
    pretrained_resnet50 = tf.keras.applications.ResNet50(
                include_top = False, input_shape = (300, 300, 3),
                pooling = 'avg', classes = 100, weights = 'imagenet',
    )

    # Here we want last 10 layers to be trainable so freezing first 40 layers
    x = 0
    for layer in pretrained_resnet50.layers:
        layer.trainable = False
        x+=1
        if x == 39:
            break

    # Adding extra Dense layers after Resnet 50 last layer,
    # we do this to increase our models capability to categorise image as having
    # fire or not
    model = Sequential()
    model.add(pretrained_resnet50)
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

    #%% Callbacks
    # Using tensorflow's learning-rate-scheduler to change learning rate at each epoch
    # this will help us to find the best learning rate for our model
    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 1e-8 * 10**(epoch/20)
    )
    # Using tensorflow's ModelCheckpoint to save best model having less validation loss
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = CHECKPOINT_FILEPATH, monitor = 'val_loss',
        save_best_only = True,
    )

    #%% Compile
    # Using Adam optimizer to optimize our model to better learn on our dataset
    model.compile(optimizer = tf.keras.optimizers.Adam(),
                loss = 'binary_crossentropy',
                metrics = 'accuracy',
    )

    #%% Train model
    # Now time to train our model on fire dataset
    model_hist = model.fit(train_set, validation_data = val_set,
            epochs = EPOCHS,
            callbacks = [learning_rate_callback, model_checkpoint_callback,],
    )

    # Save model parameters
    try:
        model.save('./fire-detection/model/fire_detection_model.keras')
    except Exception:
        model.save('./fire-detection/model/fire_detection_model.h5',
                save_format = 'h5'
        )


    #%% Graphs
    # Creating graph to visualzie how our model performed at different learning rate
    # and its loss.
    lrs = 1e-8 * (10 ** (np.arange(100) / 20))

    plt.figure(figsize=(10, 6)) # Set the figure size
    plt.grid(True) # Set the grid
    plt.semilogx(lrs, model_hist.history["loss"]) # Plot the loss in log scale
    plt.tick_config('both', length=10, width=1, which='both') # Increase tickmark size
    plt.axis([1e-8, 1e-3, 0, 1]) # Set the plot boundaries


    #%% Load model for testing
    # Downloading our best model that was picked up by Model-Checkpoint
    # best_model = tf.keras.models.load_model('/kaggle/working/final_model.h5')

    # To load model
    loaded_model = keras.models.load_model(
        './fire-detection/model/fire_detection_model.keras'
    )

    # now checking the final evalutaion of our model
    loaded_model.evaluate(val_set)

    # predefining class names so not to confuse with the output
    class_names = ['Not-Fire', 'Fire']

    # reading the input and checking the output
    pred_and_plot(loaded_model, FIRE_IMGPATH, class_names)
