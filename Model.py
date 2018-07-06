import tensorflow as tf
import numpy as np
import Net as cn
import time
import datetime
import Batch
import HandyTensorFunctions as htf

"""
For my ally is the Force, and a powerful ally it is. Life creates it, makes it
grow. Its energy surrounds us and binds us. Luminous beings are we, not this
crude matter. You must feel the Force around you; here, between you, me, the
tree, the rock, everywhere, yes. Even between the land and the ship.
"""

class Model(object):

    def __init__(self, **kwargs):

        self.sess = None
        self.name = "default_model"

        self.reset(**kwargs)


    def reset(self, **kwargs):

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.reset_op(**kwargs)

        # print("op")
        # for op in self.graph.get_operations():
        #     print(op.name)
        # print()
        # print("node")
        # for node in self.graph.as_graph_def().node:
        #     print(node.name)


    def train(self, batch, epochs, batch_size=100, display=100):

        with tf.Session(graph=self.graph) as self.sess:

            self.sess.run(tf.global_variables_initializer())
            batch.reset()

            count = 0

            while(batch.epoch_count()<epochs):

                alepsed_time = time.time()
                delta_epoch = batch.epoch_count()
                for i in range(display):
                    debug = self.train_op(batch, count)
                    count += 1

                delta_epoch = batch.epoch_count() - delta_epoch
                alepsed_time = time.time() - alepsed_time
                remaining_time = alepsed_time*(epochs-batch.epoch_count())/delta_epoch
                remaining_time = datetime.timedelta(seconds=remaining_time)
                progrssion = batch.epoch_count()/epochs

                line = "["
                bar_size = 20
                for i in range(bar_size):
                    if(i<int(bar_size*progrssion)):
                        line += "#"
                    else:
                        line += " "
                line += "] %2.1f"%(100*progrssion) + "% "

                hours, rem = divmod(remaining_time.seconds, 3600)
                minutes, seconds = divmod(rem, 60)
                line += "%2d day(s), %2dh%2dm%2ds"%(remaining_time.days,
                                                    hours,
                                                    minutes,
                                                    seconds)
                print(line, end='\r')

                #print(str(remaining_time) + " " + str(debug) + " " + str(count), end='\r')

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
        clas.set_net({"layer":"relu"})

        self.clas_input = tf.placeholder(tf.float32,
                                         shape=[None, 28, 28, 1],
                                         name="pi_input")

        self.clas_output = clas.output(self.clas_input)

        self.clas_ref_output = tf.placeholder(tf.float32,
                                              shape=[None, 10],
                                              name="pi_ref_output")

        self.clas_loss = htf.celoss(self.clas_output, self.clas_ref_output)
        self.clas_trainer = clas.trainer(self.clas_loss, tf.train.AdamOptimizer(0.0002, 0.5))

    def train_op(self, batch, count):

        batch_x, batch_y_ref = batch.train(100)
        self.sess.run(self.clas_trainer,feed_dict={self.clas_input:batch_x,
                                                   self.clas_ref_output:batch_y_ref})

        test_x, test_y_ref = batch.test(1000)
        test_y = self.clas_output.eval(session=self.sess,feed_dict={self.clas_input:test_x})

        return htf.compute_acc(test_y, test_y_ref)



model = MnistModel()
model.train(batch=Batch.MnistBatch(), epochs=10)
