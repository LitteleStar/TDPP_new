import tensorflow as tf
from tensorflow import nn
a = tf.constant([0.0, 1.0])
with tf.Session() as sess:
    b = tf.nn.softplus(a)
    print(sess.run(b))