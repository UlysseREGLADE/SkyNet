import os
import sys
import numpy as np
sys.path.append('../')
import tensorflow as tf

from SkyNet.Net import Net
from SkyNet.Model import Model
import SkyNet.HandyTensorFunctions as htf
from SkyNet.Batch.MnistBatch import MnistBatch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Classifier(Net):

    def net(self, l_l):
        #Construction du classifieur
        with tf.variable_scope("fcon1_layer"):
            l_l = self.fcon(l_l, 256)
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("fcon2_layer"):
            l_l = self.fcon(l_l, 256)
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("fcon3_layer"):
            l_l = self.fcon(l_l, 10)
            return tf.nn.softmax(l_l)


class MnistModel(Model):

    def reset_op(self, **kwargs):

        clas = Classifier('clas', self.is_training)
        clas.set_net()

        self.clas_input = tf.placeholder(tf.float32,
                                         shape=[None, 28, 28, 1],
                                         name="pi_input")

        self.clas_output = clas.output(self.clas_input)

        self.clas_ref_output = tf.placeholder(tf.float32,
                                              shape=[None, 10],
                                              name="pi_ref_output")

        self.clas_loss = htf.celoss(self.clas_output, self.clas_ref_output)
        self.clas_trainer = clas.trainer(self.clas_loss,
                                         tf.train.AdamOptimizer(0.0002, 0.5))

    def train_op(self, sess, batch, count):

        batch_x, batch_y_ref = batch.train(100)
        _, batch_y = sess.run((self.clas_trainer, self.clas_output),
                               feed_dict={self.clas_input:batch_x,
                                          self.clas_ref_output:batch_y_ref})

        test_x, test_y_ref = batch.test(1000)
        test_y = self.clas_output.eval(session=sess,
                                       feed_dict={self.clas_input:test_x})

        return {"acc_test" : htf.compute_acc(test_y, test_y_ref),
                "acc_train" : htf.compute_acc(batch_y, batch_y_ref)}



model = MnistModel(name="mnist_model")
model.train(batch=MnistBatch(), epochs=10, display=10, save=10)
