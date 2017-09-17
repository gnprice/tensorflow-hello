#!/usr/bin/env python

"""
Based on the tutorial at
https://www.tensorflow.org/get_started/get_started
"""

import tensorflow as tf

a = tf.constant(2.0)
x = tf.placeholder(tf.float32)
b = tf.Variable([-.3])
model = a * x + b

y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(model - y))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(loss, {x: [1, 2, 3, 4], y: [-1, 1, 3, 5]}))

fixb = tf.assign(b, [-3])
sess.run(fixb)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [-1, 1, 3, 5]}))
