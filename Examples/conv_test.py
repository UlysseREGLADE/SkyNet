import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

graph = tf.Graph()

with graph.as_default():
    n = tf.get_variable("n", [28, 28, 1, 1], 1)
    v = tf.get_variable("v", [1, 28, 28, 1], 1)
    r = tf.nn.conv2d(v, n, strides=[1, 1, 1, 1], padding="SAME")
    r2 = tf.nn.conv2d(v, n, strides=[1, 1, 1, 1], padding="VALID")

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    output = r.eval()
    output2 = r2.eval()
    print(output.shape, output2.shape)
