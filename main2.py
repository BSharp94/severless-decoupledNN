import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DATA_DIR  = "data"
NUM_CLASSES = 10
NUM_CHANNELS = 1
NUM_EPOCHS = 10
BATCH_SIZE = 64

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AlexModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 36, kernel_size = 2, padding=1)
        self.conv2 = nn.Conv2d(36, 36, kernel_size = 2)
        self.pooling1 = nn.AvgPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p = 0.25)

        self.conv3 = nn.Conv2d(36, 64, kernel_size = 2, padding = 1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 2)
        self.pooling2 = nn.AvgPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(p = 0.25)

        # Add Flatten in forward and backwards
        self.flatten = Flatten()
        self.linear1 = nn.Linear(7 * 7 * 64, 1028)
        self.dropout3 = nn.Dropout(p = 0.25)
        self.linear2 = nn.Linear(1028, NUM_CLASSES)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pooling1(out)
        out = self.dropout1(out)

        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.pooling2(out)
        out = self.dropout2(out)

        out = self.flatten(out)
        out = F.relu(self.linear1(out))
        out = self.dropout3(out)
        out = F.softmax(self.linear2(out))

        return out

model = AlexModel()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=BATCH_SIZE, shuffle=True)

for epoch_iter in range(NUM_EPOCHS):

    print("starting epoch ", epoch_iter + 1)

    # train?
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model.forward(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        # calc accuracy
        _, indices = out.max(1)
        acc = target.eq(indices).sum().item() / BATCH_SIZE
        print("Acc: ", acc)



# with tf.Session() as sess:

#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer()) # For local metrics - accuracy

#     for epoch_iter in range(NUM_EPOCHS):

#         # Shuffle input
#         # TODO - GET Validation set
#         shuffle_iterations = np.arange(train_x.shape[0])
#         epoch_train_x = train_x[shuffle_iterations]
#         epoch_train_y = train_y[shuffle_iterations]

#         max_iters = math.floor(train_x.shape[0] / BATCH_SIZE)
#         epoch_batch_train_x = np.split(epoch_train_x[: BATCH_SIZE * max_iters], max_iters)
#         epoch_batch_train_y = np.split(epoch_train_y[: BATCH_SIZE * max_iters], max_iters)

#         for iteration in range(math.ceil(train_x.shape[0] / BATCH_SIZE) - 1):

#             batch_x = epoch_batch_train_x[iteration]
#             batch_y = epoch_batch_train_y[iteration]
#             #epoch_train_x[iteration * BATCH_SIZE: (iteration + 1) * BATCH_SIZE]
#             #batch_y = epoch_train_y[iteration * BATCH_SIZE: (iteration + 1) * BATCH_SIZE]

#             if (batch_x.shape[0] > BATCH_SIZE - 2):
#                 _, l, a = sess.run([train_op, loss, accuracy], feed_dict={input_layer: batch_x, output_layer: batch_y})

#                 print("iter: ", iteration, " loss: ", l, " acc: ", a[0])    


#         print("Epoch complete: ", epoch_iter)

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