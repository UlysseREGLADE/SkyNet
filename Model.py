import tensorflow as tf
import numpy as np
import time
import datetime

class Model(object):

    def __init__(self, **kwargs):

        self.g = tf.Graph()
        self.sess = None
        self.name = "default_model"

        with g.as_default():
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.reset(**kwargs)

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
            debug = self.train_op(count)

            count += 1
            delta_epoch = batch.epoch_count - delta_epoch
            alepsed_time = time.time() - elapsed_time
            remaining_time = alepsed_time*batch.train_size*epochs/delta_epoch
            remaining_time = datetime.timedelta(seconds=remaining_time)

            print()

        self.sess.close()

    def train_op(self, count):
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError

    def output(i_input):
        pass

    def save(self):


class MnistModel(Model):

    def reset(self, **kwargs):

        tf.reset_default_graph()

        clas = Classifier('clas', self.is_training)
        clas.def_net(**kwargs)

        self.clas_input = tf.placeholder(tf.float32,
                                         shape=[None, 28, 28, 1],
                                         name="pi_input")

        self.clas_output = clas.output(self.clas_input)

        self.clas_ref_output = tf.placeholder(tf.float32,
                                              shape=[None, 10],
                                              name="pi_ref_output")

        self.clas_loss = htf.celoss(self.clas_output, self.clas_ref_output)
        self.clas_trainer = clas.trainer(self.clas_loss)

    def train_op():



model = MnistModel()
model.train(n=2000)
model.
