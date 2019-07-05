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
from SkyNet.Batch.SkyPix2pixBatch import SkyPix2pixBatch

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
IMAGE_SIZE = 256
LAMBDA = 100

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Discriminer(Net):

    def net(self, l_l):

        inp, tar = l_l[0], l_l[1]

        l_l = tf.concat([inp, tar], 3)

        with tf.variable_scope("down_sample_1"):
            l_l = self.conv(l_l, 64, kernel=4, strides=2)
            l_l = htf.lrelu(l_l)

        with tf.variable_scope("down_sample_2"):
            l_l = self.conv(l_l, 128, kernel=4, strides=2)
            l_l = self.batch_norm(l_l)
            l_l = htf.lrelu(l_l)

        with tf.variable_scope("down_sample_3"):
            l_l = self.conv(l_l, 256, kernel=4, strides=2)
            l_l = self.batch_norm(l_l)
            l_l = htf.lrelu(l_l)

        with tf.variable_scope("conv_4"):
            l_l = self.conv(l_l, 512, kernel=4, strides=1)
            l_l = self.batch_norm(l_l)
            l_l = htf.lrelu(l_l)

        with tf.variable_scope("conv_5"):
            l_l = self.conv(l_l, 1, kernel=4, strides=1)
            return tf.nn.sigmoid(l_l)

class Generator(Net):

    def net(self, l_l):

        channels = [64, 128, 256, 512, 512, 512, 512, 512]
        skips = []

        for i in range(len(channels)):

            with tf.variable_scope("down_sample_%02i"%(i+1)):

                l_l = self.conv(l_l, channels[i], kernel=4, strides=2)

                if(i != 0):
                    l_l = self.batch_norm(l_l)

                l_l = htf.lrelu(l_l)

                skips.append(l_l)

        for i in range(2, len(channels)+1):

            with tf.variable_scope("up_sample_%02i"%(i-1)):

                l_l = htf.unpool(l_l, 2**(i-2))
                print(i)
                print(l_l.shape)
                print(channels[-i])
                l_l = self.conv(l_l, channels[-i], kernel=4)

                l_l = self.batch_norm(l_l)

                l_l = htf.lrelu(l_l)

                l_l = tf.concat([l_l, skips[-i]], 3)

        l_l = htf.unpool(l_l, 128)
        l_l = self.conv(l_l, OUTPUT_CHANNELS, kernel=4)
        l_l = tf.nn.sigmoid(l_l)

        return l_l


class GANMnistModel(Model):

    def reset_op(self, **kwargs):

        gen = Generator('gen', self.is_training)
        gen.set_net()
        disc = Discriminer('disc', self.is_training)
        disc.set_net()

        self.gen_input = tf.placeholder(tf.float32,
                                        shape=[None, IMAGE_SIZE, IMAGE_SIZE, INPUT_CHANNELS],
                                        name="gen_input")

        self.gen_output = gen.output(self.gen_input)
        self.disc_false_input = self.gen_output

        self.disc_false_output = disc.output([self.disc_false_input, self.gen_input])

        self.disc_true_input = tf.placeholder(tf.float32,
                                              shape=[None, IMAGE_SIZE, IMAGE_SIZE, OUTPUT_CHANNELS],
                                              name="disc_true_input")

        self.disc_true_output = disc.output([self.disc_true_input, self.gen_input])

        # gradient = tf.gradients(disc_true_output_last,
        #                         [self.disc_true_input])[0]

        ones = tf.ones_like(self.disc_true_output)
        zeros = tf.zeros_like(self.disc_true_output)

        self.gen_loss = LAMBDA*htf.l1loss(self.gen_output, self.disc_true_input) + htf.ce2Dloss(self.disc_false_output, ones)
        self.disc_loss = htf.ce2Dloss(self.disc_false_output, zeros) + htf.ce2Dloss(self.disc_true_output, ones)

        self.gen_trainer = gen.trainer(self.gen_loss,
                                       tf.train.AdamOptimizer(2e-4, 0.5))
        self.disc_trainer = disc.trainer(self.disc_loss,
                                              tf.train.AdamOptimizer(2e-4, 0.5))

        self.input_list = [self.gen_input]
        self.output = self.gen_output

    def train_op(self, sess, batch, count):

        # Training data

        gen_input, disc_true_input = batch.train(100)

        # Running training

        _, _, disc_true_output, disc_false_output = sess.run((self.gen_trainer,
                                                              self.disc_trainer,
                                                              self.disc_true_output,
                                                              self.disc_false_output),
                                                             feed_dict={self.gen_input:gen_input,
                                                                        self.disc_true_input:disc_true_input})



        return {"acc_disc" : hnf.compute_acc(disc_true_output_ref,
                                             disc_true_output),
                "fal_disc" : np.mean(np.sum(disc_false_output, axis=1))}



model = GANMnistModel(name="gan_sky_pix2pix_model")
model.train(batch=SkyPix2pixBatch(), epochs=150, display=10, save=10)

with model.default_evaluator() as eval:
    gan_input = np.random.normal(0, 1, (2, 128))

    gan_output = eval.compute(gan_input)

    plt.figure()
    plt.imshow(gan_output[1, :, :, 0])
    plt.show(False)
    plt.figure()
    plt.imshow(gan_output[0, :, :, 0])
    plt.show()
