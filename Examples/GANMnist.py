import os
import sys
import numpy as np
sys.path.append('../')
import tensorflow as tf
import matplotlib.pyplot as plt

from SkyNet.Net import Net
from SkyNet.Model import Model
import SkyNet.HandyTensorFunctions as htf
import SkyNet.HandyNumpyFunctions as hnf
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
            l_l = self.fcon(l_l, 11)
            l_l = tf.nn.softmax(l_l)
            return l_l[:, :10], l_l[:, 10]

class Generator(Net):

    def net(self, l_l):
        #Construction du classifieur
        with tf.variable_scope("fcon1_layer"):
            l_l = self.fcon(l_l, 64)
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("fcon2_layer"):
            l_l = self.fcon(l_l, 128)
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("fcon3_layer"):
            l_l = self.fcon(l_l, 784)
            l_l = tf.reshape(l_l, (-1, 28, 28, 1))
            return tf.nn.sigmoid(l_l)


class GANMnistModel(Model):

    def reset_op(self, **kwargs):

        disc = Discriminer('disc', self.is_training)
        disc.set_net()
        gen = Generator('gen', self.is_training)
        gen.set_net()

        self.gen_input = tf.placeholder(tf.float32,
                                        shape=[None, 10],
                                        name="gen_input")

        self.gen_output = gen.output(self.gen_input)
        self.disc_false_input = self.gen_output

        self.disc_false_output, disc_false_output_last = disc.output(self.disc_false_input)

        gradient = tf.gradients(disc_false_output_last,
                                [self.disc_false_input])[0]

        self.disc_true_input = tf.placeholder(tf.float32,
                                              shape=[None, 28, 28, 1],
                                              name="disc_true_input")
        self.disc_true_output_ref = tf.placeholder(tf.float32,
                                                   shape=[None, 10],
                                                   name="disc_true_output_ref")
        self.disc_true_output, _ = disc.output(self.disc_true_input)

        self.gen_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(disc_false_output_last,
                                                         htf.eps, 1)))
        self.disc_false_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(disc_false_output_last,
                                                                htf.eps, 1))) + tf.reduce_sum(tf.abs(gradient))
        self.disc_true_loss = htf.celoss(self.disc_true_output,
                                         self.disc_true_output_ref) + tf.reduce_sum(tf.abs(gradient))

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



        return {"acc_disc" : hnf.compute_acc(disc_true_output_ref,
                                             disc_true_output),
                "fal_disc" : np.mean(np.sum(disc_false_output, axis=1))}



model = GANMnistModel(name="gan_mnist_model")
#model.train(batch=MnistBatch(), epochs=10, display=10, save=10)

with model.default_evaluator() as eval:
    gan_input = np.random.normal(0, 1, (2, 10))

    gan_output = eval.compute(gan_input)

    plt.figure()
    plt.imshow(gan_output[1, :, :, 0])
    plt.show(False)
    plt.figure()
    plt.imshow(gan_output[0, :, :, 0])
    plt.show()
