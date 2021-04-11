import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers, models
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten, BatchNormalization, Activation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hides tensorflow warnings

CLASSES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt' # 47 classes

def load_data():
    # loading in data
    train_data = pd.read_csv('data/emnist-balanced-train.csv')
    test_data = pd.read_csv('data/emnist-balanced-test.csv')

    y_train = train_data['45']
    y_test = test_data['41']
    x_train = train_data.drop('45',axis=1)
    x_test = test_data.drop('41',axis=1)

    return x_train, y_train, x_test, y_test

def prep_data(x_train, y_train, x_test, y_test):
    # normalizing data
    x_train /= 255.0
    x_test /= 255.0

    x_train = x_train.values.reshape(-1, 28, 28, 1) # 28x28 pixel grayscale image
    y_train = y_train.values
    x_test = x_test.values.reshape(-1, 28, 28, 1)
    y_test = y_test.values

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()
x_train, y_train, x_test, y_test = prep_data(x_train, y_train, x_test, y_test)

# model
model = keras.Sequential()
model.add(Conv2D(64, (5,5), padding='same',input_shape=(28, 28, 1), activation='relu'))

model.add(BatchNormalization())
model.add(Conv2D(64,(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(5,5),padding='same',input_shape=(28, 28, 1), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))

model.add(Conv2D(64,(5,5),padding='same',input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))

model.add(Dense(47, activation='softmax')) # 47 different outputs

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=128, epochs=4)

val_loss, val_acc = model.evaluate(x_test, y_test)
accuracy = round(val_acc, 4)
high = 0.8865
print(accuracy)

# saving model
filename = f'emnist_acc-{accuracy}'
model.save(f'/models/{filename}')



