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

        self.gen_trainer = gen.trainer(self.gen_loss,
                                       tf.train.AdamOptimizer())
        self.disc_true_trainer = disc.trainer(self.disc_true_loss,
                                              tf.train.AdamOptimizer())
        self.disc_false_trainer = disc.trainer(self.disc_false_loss,
                                               tf.train.AdamOptimizer())

        self.input_list = [self.gen_input]
        self.output = self.gen_output

    def train_op(self, sess, batch, count):

        # Training data

        disc_true_input, disc_true_output_ref = batch.train(100)
        gen_input = np.random.normal(0, 1, (100, 10))

        # Running training

        _, _, _, disc_true_output, disc_false_output = sess.run((self.gen_trainer,
                                                                 self.disc_true_trainer,
                                                                 self.disc_false_trainer,
                                                                 self.disc_true_output,
                                                                 self.disc_false_output),
                                                                 feed_dict={self.gen_input:gen_input,
                                                                            self.disc_true_input:disc_true_input,
                                                                            self.disc_true_output_ref:disc_true_output_ref})



        return {"acc_disc" : htf.compute_acc(disc_true_output_ref,
                                             disc_true_output),
                "fal_disc" : np.mean(np.sum(disc_false_output, axis=1))}



model = GANMnistModel(name="gan_mnist_model")
model.train(batch=MnistBatch(), epochs=10, display=10, save=10)
