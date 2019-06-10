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
from SkyNet.Batch.SkyBatch import SkyBatch, format_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Classifier(Net):

    def net(self, l_l):
        with tf.variable_scope("frac_fcon_1"):
            l_l = self.conv(l_l, 256, kernel=32, padding="VALID")
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("frac_fcon_2"):
            l_l = self.conv(l_l, 256, kernel=1, padding="VALID")
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("frac_fcon_3"):
            l_l = self.conv(l_l, 2, kernel=1, padding="VALID")
            return tf.nn.softmax(l_l)


class SkyModel(Model):

    def reset_op(self, **kwargs):

        clas = Classifier('clas', self.is_training)
        clas.set_net()

        self.training_input = tf.placeholder(tf.float32,
                                             shape=[None, 32, 32, 5],
                                             name="training_input")
        self.input = tf.placeholder(tf.float32,
                                    shape=[None, None, 5])

        self.reshaped_input = tf.expand_dims(self.input, 0)

        self.training_output = clas.output(self.training_input)

        self.output = clas.output(self.reshaped_input)
        self.training_output = tf.reshape(self.training_output, [-1, 2])

        self.ref_output = tf.placeholder(tf.float32,
                                         shape=[None, 2],
                                         name="pi_ref_output")

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

        return {"acc_test" : hnf.compute_acc(test_y, test_y_ref),
                "acc_train" : hnf.compute_acc(batch_y, batch_y_ref)}



batch = SkyBatch()
model = SkyModel(name="sky_model")
#model.train(batch=batch, epochs=10, display=10, save=10)

with model.default_evaluator() as eval:


    acc_table = np.zeros((batch.test_size))

    # for i in range(batch.test_size):
    for i in range(5):
        image, lab_image = batch.test_image()
        output = eval.compute(format_image(image))[0]
        acc = 1-np.mean(np.logical_xor(output[:,:,0]>0.5, lab_image>0.5))
        acc_table[i] = acc
        print(acc)


        print(np.mean(acc_table))
        plt.figure()
        plt.imshow(image)
        plt.show(False)

        plt.figure()
        plt.imshow(output[:,:,0]>0.5)
        plt.show(False)

        plt.figure()
        plt.imshow(lab_image)
        if(i==4):
            plt.show()
        else:
            plt.show(False)
