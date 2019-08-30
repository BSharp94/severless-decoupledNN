import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.losses import categorical_crossentropy
# from tensorflow.keras.optimizers import Adam

DATA_DIR  = "data"
NUM_CLASSES = 10
NUM_CHANNELS = 1
NUM_EPOCHS = 10
BATCH_SIZE = 64

# Load mnist dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data("mnist.npz")

# Normalize and add color channel
train_x = train_x / 255
train_x = np.reshape(train_x, (train_x.shape[0],train_x.shape[1], train_x.shape[2], NUM_CHANNELS))

test_x = test_x / 255
test_x = np.reshape(test_x, (test_x.shape[0],test_x.shape[1], test_x.shape[2], NUM_CHANNELS))

# One Hot Encore Values
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

IMG_WIDTH = train_x.shape[1]
IMG_HEIGHT = train_x.shape[2]

# Create Model
input_layer = tf.placeholder(tf.float32, shape = [None, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS], name = "input_layer")
conv1 = tf.layers.conv2d(input_layer, filters = 32, kernel_size = [3, 3], padding = "SAME", name = "conv1")
conv2 = tf.layers.conv2d(input_layer, filters = 32, kernel_size = [3, 3], name = "conv2")
maxpool1 = tf.layers.max_pooling2d(conv2, pool_size = [2, 2], strides = [1, 1], name = "maxpool1")
dropout1 = tf.layers.dropout(maxpool1, rate = 0.25)

conv3 = tf.layers.conv2d(dropout1, filters = 64, kernel_size=[3, 3], padding = "SAME", name = "conv3")
conv4 = tf.layers.conv2d(conv3, filters = 64, kernel_size = [3, 3], padding = "SAME", name = "conv4")
maxpool2 = tf.layers.max_pooling2d(conv4, pool_size = [2, 2], strides = [1, 1], name = "maxpool2")
dropout2 = tf.layers.dropout(maxpool2, rate = 0.25)

flatten1 = tf.layers.flatten(dropout2)
dense1 = tf.layers.dense(flatten1, units = 1028, activation=tf.nn.relu, name="dense1")
dropout3 = tf.layers.dropout(dense1, rate = 0.6)
output = tf.layers.dense(dropout3, units = NUM_CLASSES, name = "output")

output_layer = tf.placeholder(tf.int8, shape=[None, NUM_CLASSES])

loss = tf.losses.softmax_cross_entropy(output_layer, output)
optimizer = tf.train.AdamOptimizer(learning_rate=5e-4)
train_op = optimizer.minimize(loss)

# Metrics
accuracy = tf.metrics.accuracy(tf.argmax(output_layer, axis = 1), tf.argmax(output, axis = 1))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) # For local metrics - accuracy

    for epoch_iter in range(NUM_EPOCHS):

        # Shuffle input
        # TODO - GET Validation set
        shuffle_iterations = np.arange(train_x.shape[0])
        epoch_train_x = train_x[shuffle_iterations]
        epoch_train_y = train_y[shuffle_iterations]

        max_iters = math.floor(train_x.shape[0] / BATCH_SIZE)
        epoch_batch_train_x = np.split(epoch_train_x[: BATCH_SIZE * max_iters], max_iters)
        epoch_batch_train_y = np.split(epoch_train_y[: BATCH_SIZE * max_iters], max_iters)

        for iteration in range(math.ceil(train_x.shape[0] / BATCH_SIZE) - 1):

            batch_x = epoch_batch_train_x[iteration]
            batch_y = epoch_batch_train_y[iteration]
            #epoch_train_x[iteration * BATCH_SIZE: (iteration + 1) * BATCH_SIZE]
            #batch_y = epoch_train_y[iteration * BATCH_SIZE: (iteration + 1) * BATCH_SIZE]

            if (batch_x.shape[0] > BATCH_SIZE - 2):
                _, l, a = sess.run([train_op, loss, accuracy], feed_dict={input_layer: batch_x, output_layer: batch_y})

                print("iter: ", iteration, " loss: ", l, " acc: ", a[0])    


        print("Epoch complete: ", epoch_iter)

# model = Sequential()

# # Section 1
# model.add(Conv2D(32, kernel_size = (3, 3), padding="same", activation="relu", input_shape = train_x.shape[1:]))
# model.add(Conv2D(32, kernel_size = (3, 3), activation="relu"))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# # Section 2
# model.add(Conv2D(64, kernel_size = (3, 3), padding = "same", activation = "relu"))
# model.add(Conv2D(64, kernel_size = (3, 3), activation="relu"))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# # Section 3
# model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.6))
# model.add(Dense(NUM_CLASSES, activation="softmax"))

# model.compile(loss = categorical_crossentropy, optimizer=Adam(lr=5e-4), metrics = ["accuracy"])

# model.fit(train_x, train_y, batch_size=64, epochs=10, validation_split=0.1, verbose=1)

# score = model.evaluate(test_x, test_y)
# print("test loss: ", score[0])
# print("test acc: ", score[1])


print("complete")