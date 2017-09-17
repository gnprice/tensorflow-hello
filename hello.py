#!/usr/bin/env python

"""
Based on the tutorial at
https://www.tensorflow.org/get_started/get_started
"""

import os
# TODO provision a properly optimized version of TF -- I've been getting
# W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
# and ditto for SSE4.2, AVX, AVX2, FMA.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
