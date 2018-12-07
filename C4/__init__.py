import tensorflow as tf

tf.enable_eager_execution()

a = tf.constant([[1, 2],
                 [3, 4]])

b = tf.add(a, 1)
print(b)

