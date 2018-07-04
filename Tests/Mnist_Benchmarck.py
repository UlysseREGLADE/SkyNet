# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 23:10:00 2018

@author: Ulysse
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

import Net as cn
import HandyTensorFunctions as htf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

class Classifier(cn.Net):
    
    def set_net(self, param = {}):
        if("layer" in param):
            self.layer = param["layer"]
        else:
            self.layer = "lrelu"

    def net(self, l_l):
        #Construction du classifieur
        with tf.variable_scope("layer_2"):
            l_l = self.fcon(l_l, 100)
            if("bn_" in self.layer):
                l_l = self.normalize(l_l)
            if("lrelu" in self.layer):
                l_l = htf.lrelu(l_l)
            elif("relu" in self.layer):
                l_l = tf.nn.relu(l_l)
            if("_bn" in self.layer):
                l_l = self.normalize(l_l)
        with tf.variable_scope("layer_3"):
            l_l = self.fcon(l_l, 100)
            if("bn_" in self.layer):
                l_l = self.normalize(l_l)
            if("lrelu" in self.layer):
                l_l = htf.lrelu(l_l)
            elif("relu" in self.layer):
                l_l = tf.nn.relu(l_l)
            if("_bn" in self.layer):
                l_l = self.normalize(l_l)
        with tf.variable_scope("layer_4"):
            if("bn_" in self.layer):
                l_l = self.normalize(l_l)
            l_l = self.fcon(l_l, 10)
            if("_bn" in self.layer):
                l_l = self.normalize(l_l)
            return tf.nn.softmax(l_l)

class Model():
    def __init__(self):
        pass
    def reset(self, param = {}):
        #On commence par reset le graph de tensorflow
        ops.reset_default_graph()
        #Puis on construit le graph de calcule
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        clas = Classifier('clas', self.is_training)
        clas.set_net(param)

        self.clas_input = tf.placeholder(tf.float32,
                                    shape=[None, 28, 28, 1],
                                    name="pi_input")

        self.clas_output = clas.output(self.clas_input)

        self.clas_ref_output = tf.placeholder(tf.float32,
                                         shape=[None, 10],
                                         name="pi_ref_output")

        self.clas_loss = htf.celoss(self.clas_output, self.clas_ref_output)
        self.clas_trainer = clas.trainer(self.clas_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, N):

        history = np.zeros((N, 2))
        history[:,0] = np.arange(N)

        for i in range(N):
            x_train, y_train = mnist.train.next_batch(100)
            x_train = x_train.reshape((100, 28, 28, 1))

            self.sess.run(self.clas_trainer,
                          feed_dict={self.clas_input: x_train,
                                     self.clas_ref_output: y_train,
                                     self.is_training:True})

            T=2000

            index = np.arange(T)
            x_train,y_train=mnist.test.images[index,:],mnist.test.labels[index,:]
            x_train = x_train.reshape((T, 28, 28, 1))

            vect=self.clas_output.eval(session=self.sess,
                                       feed_dict={self.clas_input: x_train,
                                                  self.is_training:False})
            precision=np.sum((np.argmax(vect,axis=1)==np.argmax(y_train,axis=1)))/T
            history[i, 1] = precision

            print(i, history[i, 1])

        return history


model = Model()
params = [{"layer":"relu"},
          {"layer":"lrelu"},
          {"layer":"bn_relu"},
          {"layer":"relu_bn"},
          {"layer":"bn_lrelu"},
          {"layer":"lrelu_bn"}]

plt.figure()
for param in params:
    model.reset(param)
    history = model.train(2000)
    plt.plot(history[:,0], history[:,1], label=str(param))
plt.legend()
plt.show()
