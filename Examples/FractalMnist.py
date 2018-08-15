import os
import sys
import numpy as np
sys.path.append('../')
import tensorflow as tf

from SkyNet.Net import Net
from SkyNet.Model import Model
import SkyNet.HandyTensorFunctions as htf
from SkyNet.Batch.MnistBatch import MnistBatch

class FractalClassifier(Net):

    def net(self, l_l):

        with tf.variable_scope("frac_fcon_1"):
            l_l = self.conv(l_l, 256, kernel=28, padding="VALID")
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("frac_fcon_2"):
            l_l = self.conv(l_l, 256, kernel=1, padding="VALID")
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("frac_fcon_3"):
            l_l = self.conv(l_l, 10, kernel=1, padding="VALID")
            return tf.nn.softmax(l_l)

class FractalMnistModel(Model):

    def reset_op(self, **kwargs):

        clas = FractalClassifier('clas', self.is_training)
        clas.set_net()

        self.input = tf.placeholder(tf.float32,
                                    shape=[None, None, None, 1],
                                    name="input")

        self.training_input = tf.placeholder(tf.float32,
                                             shape=[None, 28, 28, 1],
                                             name="training_input")

        self.training_output = clas.output(self.training_input)
        self.training_output = tf.reshape(self.training_output, [-1, 10])

        self.output = clas.output(self.input)

        self.ref_output = tf.placeholder(tf.float32,
                                         shape=[None, 10],
                                         name="ref_output")

        self.loss = htf.celoss(self.training_output, self.ref_output)
        self.trainer = clas.trainer(self.loss,
                                    tf.train.AdamOptimizer(0.0002, 0.5))

        self.input_list = [self.input]

    def train_op(self, sess, batch, count):

        batch_x, batch_y_ref = batch.train(100)
        _, batch_y = sess.run((self.trainer, self.training_output),
                               feed_dict={self.training_input:batch_x,
                                          self.ref_output:batch_y_ref})

        test_x, test_y_ref = batch.test(1000)
        test_y = self.training_output.eval(session=sess,
                                           feed_dict={self.training_input:test_x})

        return {"acc_test" : htf.compute_acc(test_y, test_y_ref),
                "acc_train" : htf.compute_acc(batch_y, batch_y_ref)}

model = FractalMnistModel(name="fractal_mnist_model")
model.train(batch=MnistBatch(), epochs=10, display=10, save=10)

with model.default_evaluator() as eval:
    eval.compute( np.zeros((1,28,28,1)) )
