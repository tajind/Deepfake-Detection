# imports

import pathlib

import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import RandomZoom, RandomRotation
from keras.models import Sequential
from keras.optimizer_v1 import SGD
from matplotlib import pyplot as plt

import utils
from classifiers import *
from utils import *
import tensorflow as tf

# setting tensorflow variables and pritint tensorflow information
tf.get_logger().setLevel('ERROR')
print('TensorFlow version: ', tf.__version__)
print(f"Num of GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# dataset paths
dataset_path = pathlib.Path('real-vs-fake')
train_path = pathlib.Path('real-vs-fake/train')
val_path = pathlib.Path('real-vs-fake/valid')

batch_size = 32
IMG_SIZE = (224, 224)


# dataset generate function
def dataset():
    train_ = tf.keras.utils.image_dataset_from_directory(
        train_path,
        subset="training",
        validation_split=0.2,
        seed=100,
        image_size=IMG_SIZE,
        batch_size=batch_size)

    val_ = tf.keras.utils.image_dataset_from_directory(
        val_path,
        subset="validation",
        validation_split=0.2,
        seed=100,
        image_size=IMG_SIZE,
        batch_size=batch_size)

    return train_, val_

# normalize the incoming data from the dataset function
def normalize(train):
    # standardize the dataset images
    norm_layer = layers.Rescaling(1. / 255)
    normalized_ = train.map(lambda x, y: (norm_layer(x), y))
    return normalized_


# trains the model on meso4
def train_meso4(epochs):
    # get the dataset
    train_, val_ = dataset()

    # normalize the datatset
    train_ = normalize(train_)

    # model definition
    model = Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        layers.Conv2D(8, (5, 5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        layers.Conv2D(16, (5, 5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        layers.Conv2D(16, (5, 5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(4, 4), padding='same'),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(16),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    # Adam
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    # compile the model with the Adam optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    print(len(model.layers))

    # # SGD
    # learning_rate = 0.001
    # optimizer = SGD(lr=learning_rate, momentum=0.01)
    # model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    # model.summary()

    # get the model traning history
    history = model.fit(
        train_,
        validation_data=val_,
        epochs=epochs
    )

    # save the wights of the model to path
    model.save_weights(f'Meso4_New{epochs}_224_224.h5')

    # plot the history
    utils.his_ploter("Meso4", history, epochs)

# model definition function - returns the model
def AlexNet(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(96, (11, 11), strides=4, name="conv0")(X_input)
    X = BatchNormalization(axis=3, name="bn0")(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max0')(X)

    X = Conv2D(256, (5, 5), padding='same', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max1')(X)

    X = Conv2D(384, (3, 3), padding='same', name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    X = Conv2D(384, (3, 3), padding='same', name='conv3')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)

    X = Conv2D(256, (3, 3), padding='same', name='conv4')(X)
    X = BatchNormalization(axis=3, name='bn4')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max2')(X)

    X = Flatten()(X)

    X = Dense(4096, activation='relu', name="fc0")(X)
    X = Dropout(0.5)(X)
    X = Dense(4096, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)
    X = Dense(1, activation='softmax', name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name='AlexNet')
    return model

# trains the model on alexnet
def train_alexnet(epochs):
    # get the dataset
    train_, val_ = dataset()

    # normalize the datatset
    train_ = normalize(train_)

    model = AlexNet((224, 224, 3))

    # compile the model with the Adam optimizer
    model.compile(optimizer=Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # get the model traning history
    history = model.fit(
        train_,
        validation_data=val_,
        epochs=epochs
    )

    # save the wights of the model to path
    model.save_weights(f'AlexNet_New{epochs}.h5')

    # plot the history
    his_ploter("AlexNet", history, epochs)


# plots the history on the graph
def his_ploter(name, history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Training and Validation Loss - {name}')
    plt.show()
