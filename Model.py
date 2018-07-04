import tensorflow as tf
import numpy as np
import Net as cn
import time
import datetime
import Batch

class Model(object):

    def __init__(self, **kwargs):

        self.sess = None
        self.name = "default_model"

        self.reset(**kwargs)


    def reset(self, **kwargs):

        self.g = tf.Graph()
        with self.g.as_default():
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.reset_op(**kwargs)


    def train(self, batch, epochs):

        if(self.sess is not None):
            self.sess.close()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        batch.reset()
        count = 0

        while(batch.epoch_count()<batch.train_size*epochs):

            alepsed_time = time.time()
            delta_epoch = batch.epoch_count
            debug = self.train_op(batch, count)

            count += 1
            delta_epoch = batch.epoch_count - delta_epoch
            alepsed_time = time.time() - elapsed_time
            remaining_time = alepsed_time*batch.train_size*epochs/delta_epoch
            remaining_time = datetime.timedelta(seconds=remaining_time)

            print(remaining_time)

        self.sess.close()

    def train_op(self, count):
        raise NotImplementedError

    def reset_op(self, **kwargs):
        raise NotImplementedError

    def output(i_input):
        pass


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


class MnistModel(Model):

    def reset_op(self, **kwargs):

        clas = Classifier('clas', self.is_training)
        clas.set_net(**kwargs)

        self.clas_input = tf.placeholder(tf.float32,
                                         shape=[None, 28, 28, 1],
                                         name="pi_input")

        self.clas_output = clas.output(self.clas_input)

        self.clas_ref_output = tf.placeholder(tf.float32,
                                              shape=[None, 10],
                                              name="pi_ref_output")

        self.clas_loss = htf.celoss(self.clas_output, self.clas_ref_output)
        self.clas_trainer = clas.trainer(self.clas_loss)

    def train_op(batch, count):
        batch_x, batch_y = batch.train(100)
        self.sess.run(self.clas_trainer, feed_dict={self.clas_input:batch_x,
                                                    self.clas_ref_output:batch_y})
        return self.clas_loss.eval(session=self.sess, feed_dict={self.clas_input:batch_x,
                                                                 self.clas_ref_output:batch_y})



model = MnistModel()
model.train(batch=Batch.MnistBatch(), epochs=10)
