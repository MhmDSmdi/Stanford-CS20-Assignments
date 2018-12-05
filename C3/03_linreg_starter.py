import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = 'birth_life_2010.txt'

data, n_samples = utils.read_birth_life_data(DATA_FILE)
print(data)
X, Y = None, None
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

w, b = None, None
w = tf.Variable(name="w", initial_value=0.0)
b = tf.Variable(name="b", initial_value=0.0)

Y_predicted = None
Y_predicted = w * X + b

loss = None
loss = tf.square(Y - Y_predicted, name="loss")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()

with tf.Session() as sess:
    sess.run(w.initializer)
    sess.run(b.initializer)

    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += loss_

        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    w_out, b_out = None, None
    w_out = sess.run(w)
    b_out = sess.run(b)

print('Took: %f seconds' % (time.time() - start))

plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()
