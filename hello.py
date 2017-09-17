#!/usr/bin/env python

"""
Based on the tutorial at
https://www.tensorflow.org/get_started/get_started
"""

import tensorflow as tf

## Graph

a = tf.constant(2.0)
x = tf.placeholder(tf.float32)
b = tf.Variable([-.3])
model = a * x + b
with tf.name_scope('model'):
    tf.summary.scalar('bias', tf.reduce_mean(b))

y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(model - y))
tf.summary.scalar('loss', loss)

merged = tf.summary.merge_all()

## Data

data = {x: [1, 2, 3, 4], y: [-1, 1, 3, 5]}

## Session

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print("Initial:", sess.run(loss, data))

fixb = tf.assign(b, [-3])
sess.run(fixb)
print("Cheating:", sess.run(loss, data))
sess.run(init)

writer = tf.summary.FileWriter('log', sess.graph)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(1000):
    sess.run(train, data)
    if (i+1) % 10 == 0:
        summary = sess.run(merged, data)
        writer.add_summary(summary, i)
b_tr, loss_tr = sess.run([b, loss], data)
print("Training: b %s, loss %s" % (b_tr, loss_tr))

print("""
Now try:
  tensorboard --logdir=log
(You might want to remove data from any previous runs.)
""")
