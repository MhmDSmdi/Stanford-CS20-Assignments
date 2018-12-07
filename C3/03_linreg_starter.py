import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DATA_FILE = 'birth_life_2010.txt'

data, n_samples = utils.read_birth_life_data(DATA_FILE)
X, Y = None, None
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

w2, b2, w, b = None, None, None, None
w = tf.get_variable("weight", initializer=tf.constant(0.0))
b = tf.get_variable("bias", initializer=tf.constant(0.0))
w2 = tf.get_variable("huber_weight", initializer=tf.constant(0.0))
b2 = tf.get_variable("huber_bias", initializer=tf.constant(0.0))

Y_predicted2, Y_predicted = None, None
Y_predicted = w * X + b
Y_predicted2 = w2 * X + b2


def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)

    def f1(): return 0.5 * tf.square(residual)

    def f2(): return delta * residual - 0.5 * tf.square(delta)

    return tf.cond(residual < delta, f1, f2)


loss, h_loss = None, None
loss = tf.square(Y - Y_predicted, name="loss")
h_loss = huber_loss(Y_predicted2, Y, 14.0)

optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(h_loss)

start = time.time()

with tf.Session() as sess:
    sess.run(w.initializer)
    sess.run(b.initializer)
    sess.run(w2.initializer)
    sess.run(b2.initializer)
    for i in range(100):
        total_loss1 = 0
        total_loss2 = 0
        for x, y in data:
            _, loss_ = sess.run([optimizer1, loss], feed_dict={X: x, Y: y})
            _, h_loss_ = sess.run([optimizer2, h_loss], feed_dict={X: x, Y: y})
            total_loss1 += loss_
            total_loss2 += h_loss_
        print('Epoch {0}: {1} => MSE loss function'.format(i, total_loss1 / n_samples))
        print('Epoch {0}: {1} => Huber loss function'.format(i, total_loss2 / n_samples))

    w_out, b_out, w_out, b_out = None, None, None, None
    w_out = sess.run(w)
    b_out = sess.run(b)
    w2_out = sess.run(w2)
    b2_out = sess.run(b2)

print('Took: %f seconds' % (time.time() - start))

plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
plt.plot(data[:, 0], data[:, 0] * w_out + b_out, 'r', label='Predicted data Using MSE loss')
plt.plot(data[:, 0], data[:, 0] * w2_out + b2_out, 'g', label='Predicted data Using Huber loss')
plt.legend()
plt.show()
