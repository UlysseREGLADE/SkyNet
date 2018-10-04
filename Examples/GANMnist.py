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

class Discriminer(Net):

    def net(self, l_l):
        #Construction du classifieur
        with tf.variable_scope("fcon1_layer"):
            l_l = self.fcon(l_l, 128)
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("fcon2_layer"):
            l_l = self.fcon(l_l, 64)
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("fcon3_layer"):
            l_l = self.fcon(l_l, 10)
            return tf.nn.softmax(l_l)

class Genrater(Net):

    def net(self, l_l):
        #Construction du classifieur
        with tf.variable_scope("fcon1_layer"):
            l_l = self.fcon(l_l, 64)
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("fcon2_layer"):
            l_l = self.fcon(l_l, 128)
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("fcon3_layer"):
            l_l = self.fcon(l_l, 10)
            return tf.nn.softmax(l_l)


class GANMnistModel(Model):

    def reset_op(self, **kwargs):

        disc = Discriminer('disc', self.is_training)
        disc.set_net()
        gen = Generater('gen', self.is_training)
        gen.set_net()

        self.gen_input = tf.placeholder(tf.float32,
                                        shape=[None, 10],
                                        name="gen_input")

        self.gen_output = gen.output(self.gen_input)
        self.disc_false_input = self.gen_output

        self.disc_false_output = self.disc(self.disc_false_input)

        self.disc_true_input = tf.placeholder(tf.float32,
                                              shape=[None, 28, 28, 1],
                                              name="disc_true_input")
        self.disc_true_output_ref = tf.placeholder(tf.float32,
                                                   shape=[None, 10],
                                                   name="disc_true_output_ref")
        self.disc_true_output = self.disc(self.disc_true_input)

        self.gen_loss = tf.mean(-tf.log(tf.clip_by_value(1-tf.sum(self.disc_false_output,
                                                                  axis=1),
                                                         htf.eps, 1)))
        self.disc_false_loss = tf.mean(-tf.log(tf.clip_by_value(tf.sum(self.disc_false_output,
                                                                       axis=1),
                                                                htf.eps, 1)))
        self.disc_true_loss = htf.celoss(self.disc_true_output,
                                         self.disc_true_output_ref)

        self.loss = htf.celoss(self.output, self.ref_output)
        self.trainer = clas.trainer(self.loss,
                                    tf.train.AdamOptimizer(0.0002, 0.5))

        self.input_list = [self.input]

    def train_op(self, sess, batch, count):

        batch_x, batch_y_ref = batch.train(100)
        _, batch_y = sess.run((self.trainer, self.output),
                               feed_dict={self.input:batch_x,
                                          self.ref_output:batch_y_ref})

        test_x, test_y_ref = batch.test(1000)
        test_y = self.output.eval(session=sess,
                                  feed_dict={self.input:test_x})

        return {"acc_test" : htf.compute_acc(test_y, test_y_ref),
                "acc_train" : htf.compute_acc(batch_y, batch_y_ref)}



model = MnistModel(name="mnist_model")
model.train(batch=MnistBatch(), epochs=10, display=10, save=10)

with model.default_evaluator() as eval:
    eval.compute( np.zeros((1,28,28,1)) )
