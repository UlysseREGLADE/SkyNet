import os
import sys
import numpy as np
sys.path.append('../')
import tensorflow as tf
import matplotlib.pyplot as plt

from SkyNet.Net import Net
from SkyNet.Model import Model
import SkyNet.HandyTensorFunctions as htf
from SkyNet.Batch.MnistBatch import MnistBatch
from SkyNet.Batch.FractalBatch import FractalBatch

class FractalClassifier(Net):

    def net(self, l_l):

        with tf.variable_scope("frac_fcon_1"):
            l_l = self.conv(l_l, 256, kernel=32, padding="VALID")
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("frac_fcon_2"):
            l_l = self.conv(l_l, 256, kernel=1, padding="VALID")
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("frac_fcon_3"):
            l_l = self.conv(l_l, 11, kernel=1, padding="VALID")
            return tf.nn.softmax(l_l)

class FractalMnistModel(Model):

    def reset_op(self, **kwargs):

        clas = FractalClassifier('clas', self.is_training)
        clas.set_net()

        self.input = tf.placeholder(tf.float32,
                                    shape=[None, None, 1],
                                    name="input")
        self.reshaped_input = tf.expand_dims(self.input, 0)

        self.training_input = tf.placeholder(tf.float32,
                                             shape=[None, 32, 32, 1],
                                             name="training_input")

        self.training_output = clas.output(self.training_input)
        self.training_output = tf.reshape(self.training_output, [-1, 11])

        self.output = clas.output(self.reshaped_input)
        self.output = tf.squeeze(self.output, [0])

        self.ref_output = tf.placeholder(tf.float32,
                                         shape=[None, 11],
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

batch = FractalBatch(parent_batch=MnistBatch, added_crop=4)
model = FractalMnistModel(name="fractal_mnist_model")
#model.train(batch=batch, epochs=10, display=10, save=10)

def generate_image(batch=batch.parent_batch, shape=(256, 256), numbers=6):

    image = np.zeros(list(shape) + [1])
    patchs, _ = batch.test(numbers)

    for i in range(numbers):
        y_shift = np.random.randint(shape[0] - batch.input_shape[0])
        x_shift = np.random.randint(shape[1] - batch.input_shape[1])
        image[y_shift:y_shift+batch.input_shape[0], x_shift:x_shift+batch.input_shape[1], :] = patchs[i]

    return image

np.random.seed(23456745)
colors = np.ones((11, 4))
colors[:10, :3] = np.random.rand(10, 3)
colors[10, :3] = 0

with model.default_evaluator() as eval:

    image = generate_image()
    plot = np.zeros((image.shape[0], image.shape[1], 4))

    res = eval.compute(image)
    res[:, :, 10] = (1 - res[:, :, 10])*0.8
    output = np.einsum('ijk,kl', res, colors)
    plot[16:-15, 16:-15, :] = output

    fig = plt.figure(frameon=False)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.imshow(plot)

    for i in range(10):
        plt.text(12+12*i, 18, str(i), color=colors[i], fontsize=12)

    plt.show()
