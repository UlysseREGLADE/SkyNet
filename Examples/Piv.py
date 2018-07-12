import tensorflow as tf
import numpy as np

import os
import sys
sys.path.append('../')
from SkyNet.Net import Net
from SkyNet.Model import Model
from SkyNet.Batch.PivBatch import PivBatch
import SkyNet.HandyTensorFunctions as htf

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class Classifier(Net):

    def set_net(self, param = {}):
        if("layer" in param):
            self.layer = param["layer"]
        else:
            self.layer = "lrelu"

    def net(self, l_l):
        #Construction du classifieur
        with tf.variable_scope("conv1_layer"):
            l_l = self.conv(l_l, 64, kernel=5)
            l_l = tf.nn.relu(l_l)
            l_l = htf.pool(l_l)
            #l_l = tf.nn.lrn(l_l, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')

        with tf.variable_scope("conv2_layer"):
            l_l = self.conv(l_l, 64, kernel=5)
            l_l = tf.nn.relu(l_l)
            l_l = htf.pool(l_l)
            #l_l = tf.nn.lrn(l_l, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')

        with tf.variable_scope("fcon1_layer"):
            l_l = self.fcon(l_l, 256)
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("fcon2_layer"):
            l_l = self.fcon(l_l, 6)
            return l_l


class PIVModel(Model):

    def reset_op(self, **kwargs):

        clas = Classifier('clas', self.is_training)
        clas.set_net()

        self.clas_input = tf.placeholder(tf.float32,
                                         shape=[None, 48, 48, 2],
                                         name="pi_input")

        self.clas_output = clas.output(self.clas_input)

        self.clas_ref_output = tf.placeholder(tf.float32,
                                              shape=[None, 6],
                                              name="pi_ref_output")

        self.clas_loss = htf.l1loss(self.clas_output, self.clas_ref_output)
        self.clas_trainer = clas.trainer(self.clas_loss,
                                         tf.train.AdamOptimizer(0.0002, 0.5))

    def train_op(self, sess, batch):

        batch_x, batch_y_ref = batch.train(100)
        _, batch_loss = sess.run((self.clas_trainer, self.clas_loss),
                              feed_dict={self.clas_input:batch_x,
                                         self.clas_ref_output:batch_y_ref})

        test_x, test_y_ref = batch.test(1000)
        test_loss = self.clas_loss.eval(session=sess,
                                       feed_dict={self.clas_input:test_x,
                                                  self.clas_ref_output:test_y_ref})

        return {"loss_test" : test_loss,
                "loss_train" : batch_loss}



model = PIVModel()
model.train(batch=PivBatch(), epochs=1, display=10, save=100)
