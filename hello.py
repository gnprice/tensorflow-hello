#!/usr/bin/env python

"""
From the tutorial at
https://www.tensorflow.org/get_started/get_started
"""

import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # implicitly also float32
node3 = tf.add(node1, node2)  # or `node1 + node2` as a shortcut

sess = tf.Session()

print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
