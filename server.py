import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

model_1 = tf.keras.models.load_model('model_1')
model_2 = tf.keras.models.load_model('model_2')

w1 = model_1.get_weights()
w2 = model_2.get_weights()

# print('this is w1: ', len(w1[0]), len(w1[0][0]))

w = []
for i in range(0, len(w1)):
    w.append(np.array((w1[i] + w2[i]) / 2))

# print('this is new w: ', len(w[0]), len(w[0][0]))
new_model = model_1
new_model.set_weights(w)

new_model.save('new_model')
# model_1 = tf.keras.models.load_model('model_1_new')
# w = model_1.get_weights()
# print(w)






