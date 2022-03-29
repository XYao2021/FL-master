import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras.callbacks import TensorBoard

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_1_train = []
y_1_train = []

x_2_train = []
y_2_train = []

for i in range(0, len(y_train)):
    if y_train[i] <= 2:
        x_1_train.append(x_train[i])
        y_1_train.append(y_train[i])

    if 6 >= y_train[i] >= 4:
        x_2_train.append(x_train[i])
        y_2_train.append(y_train[i])

# print(len(x_1_train),len(x_2_train))
# print(np.array(x_0_train[0]))
# plt.imshow(x_1_train[8000], cmap=plt.cm.binary)
# plt.show()

x_1_train = tf.keras.utils.normalize(x_1_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

x_2_train = tf.keras.utils.normalize(x_2_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
"-------------------------User_1 Training------------------------------"
# Name_1 = "training-model-1-{}".format(int(time.time()))
#
# tensorboard = TensorBoard(log_dir='logs/{}'.format(Name_1))

model_1 = tf.keras.models.Sequential()
model_1.add(tf.keras.layers.Flatten())  # input layer
model_1.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # dense layer
model_1.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # dense layer
model_1.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer

model_1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_1.fit(x_1_train, np.array(y_1_train), epochs=6)
# weight_1 = model_1.get_weights()
# model_1.save_weights('w1.model')
model_1.save('model_1')

"-------------------------User_2 Training------------------------------"
# Name_2 = "training-model-2-{}".format(int(time.time()))
#
# tensorboard = TensorBoard(log_dir='logs/{}'.format(Name_2))

model_2 = tf.keras.models.Sequential()
model_2.add(tf.keras.layers.Flatten())  # input layer
model_2.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # dense layer
model_2.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # dense layer
model_2.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer

model_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_2.fit(x_2_train, np.array(y_2_train), epochs=6)
# weight_2 = model_2.get_weights()

# model_2.save_weights('w2.model')
model_2.save('model_2')
# lines = [str(weight_1), str(weight_2)]
# with open('training_w.txt', 'w') as f:
#     for line in lines:
#         f.write(line)
#         f.write('\n')
# print(len(weight_2), len(weight_2[0]), '\n')
# print(weight_1[0], '\n', weight_2[0])
#
# val_loss, val_acc = model.evaluate(x_test, y_test)
# # print(val_loss, val_acc)
#
# model.save('epic_num_reader.model')
#
# new_model = tf.keras.models.load_model('epic_num_reader.model')
# print(new_model)
# print(model)
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()
# print(x_train[0])


