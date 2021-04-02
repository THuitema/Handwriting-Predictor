import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten, BatchNormalization, Activation
import matplotlib.pyplot as plt
import ssl
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hides tensorflow warnings

# handles weird errors when loading dataset from keras
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

    
def load_data():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # adding 1 at end to add dimension for just gray-scale values
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) 
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    return x_train, y_train, x_test, y_test

def prep_data(x_train, x_test):
    x_train_normalized = x_train.astype('float32')
    x_test_normalized = x_test.astype('float32')
    
    # Normalizing values between 0 and 1
    x_train_normalized = keras.utils.normalize(x_train_normalized, axis=1)
    x_test_normalized = keras.utils.normalize(x_test_normalized, axis=1)
    return x_train_normalized, x_test_normalized

x_train, y_train, x_test, y_test = load_data()
x_train, x_test = prep_data(x_train, x_test)

# model = keras.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
#     layers.MaxPool2D((2,2)), # gets max pixel in every 2x2 grid
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#
#     layers.BatchNormalization(),
#     layers.Flatten(), # flattens 2D array to 1D
#
#     # deepening model with more layers
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(128, activation='relu'),
#
#     layers.Dense(10, activation='softmax') # output layer, 10 = number of outputs
# ])
model= models.Sequential()
model.add(Conv2D(28,(5,5),padding='same',input_shape=(28, 28, 1), activation='relu'))

model.add(BatchNormalization())
model.add(Conv2D(28,(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32,(5,5),padding='same',input_shape=(28, 28, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))


model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=128, epochs=8)

val_loss, val_acc = model.evaluate(x_test, y_test)

# saving model if val_acc > most accurate model
high = .9890
if val_acc > high:
    model.save(f'models/acc-{round(val_acc, 4)}')
    print('model saved')

# loading model and showing predictions from x_test
loaded = keras.models.load_model('models/acc-0.9942')
predictions = loaded.predict([x_test])

for count, i in enumerate(predictions):
    print(i, y_test[count])
    plt.imshow(x_test[count])
    plt.title(f'Predicted: {np.argmax(i)}, Actual: {y_test[count]}')
    plt.show()
