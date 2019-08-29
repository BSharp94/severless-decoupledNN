import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

DATA_DIR  = "data"
NUM_CLASSES = 10
NUM_CHANNELS = 1

# Load mnist dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data("mnist.npz")

# Normalize and add color channel
train_x = train_x / 25
train_x = np.reshape(train_x, (train_x.shape[0],train_x.shape[1], train_x.shape[2], NUM_CHANNELS))

test_x = test_x / 25
test_x = np.reshape(test_x, (test_x.shape[0],test_x.shape[1], test_x.shape[2], NUM_CHANNELS))

# One Hot Encore Values
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

# Create Model
model = Sequential()

# Section 1
model.add(Conv2D(32, kernel_size = (3, 3), padding="same", activation="relu", input_shape = train_x.shape[1:]))
model.add(Conv2D(32, kernel_size = (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Section 2
model.add(Conv2D(64, kernel_size = (3, 3), padding = "same", activation = "relu"))
model.add(Conv2D(64, kernel_size = (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Section 3
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(NUM_CLASSES, activation="softmax"))

model.compile(loss = categorical_crossentropy, optimizer=Adam(lr=5e-4), metrics = ["accuracy"])

model.fit(train_x, train_y, batch_size=64, epochs=10, validation_split=0.1, verbose=1)

score = model.evaluate(test_x, test_y)
print("test loss: ", score[0])
print("test acc: ", score[1])


print("complete")