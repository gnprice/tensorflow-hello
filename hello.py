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

data = {x: [1, 2, 3, 4], y: [-1, 1, 3, 5]}

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print("Initial:", sess.run(loss, data))

fixb = tf.assign(b, [-3])
sess.run(fixb)
print("Cheating:", sess.run(loss, data))
sess.run(init)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(1000):
    sess.run(train, data)
b_tr, loss_tr = sess.run([b, loss], data)
print("Training: b %s, loss %s" % (b_tr, loss_tr))
