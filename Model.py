import tensorflow as tf
import numpy as np
import Net as cn
import time
import datetime
from Batch.Cifar10Batch import Cifar10Batch
import HandyTensorFunctions as htf

"""
For my ally is the Force, and a powerful ally it is. Life creates it, makes it
grow. Its energy surrounds us and binds us. Luminous beings are we, not this
crude matter. You must feel the Force around you; here, between you, me, the
tree, the rock, everywhere, yes. Even between the land and the ship.
"""

# print("op")
# for op in self.graph.get_operations():
#     print(op.name)
# print()
# print("node")
# for node in self.graph.as_graph_def().node:
#     print(node.name)

class Model(object):

    def __init__(self, **kwargs):

        self.name = "default_model"

        self.reset(**kwargs)


    def reset(self, **kwargs):

        self.graph = tf.Graph()
        self.graph_param = kwargs

        if("name" in kwargs):
            self.name = kwargs["name"]
        else:
            self.name = "default_model"

        with self.graph.as_default():
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.reset_op(**kwargs)

        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self.sess_state = {}
        for trainable in trainables:
            self.sess_state[trainable.name] = (trainable, None)

    def train(self, batch, epochs, batch_size=100, display=100):

        with tf.Session(graph=self.graph) as sess:

            #Affichage du nom du graph entrainne
            print("\nStarting training:")
            print("Graph name: " + str(self.name))

            #Initialisation de la session
            sess.run(tf.global_variables_initializer())
            batch.reset()

            #Boucle d'entrainnement
            count = 0
            while(batch.epoch_count()<epochs):

                #On effectue le trainning
                alepsed_time = time.time()
                delta_epoch = batch.epoch_count()
                for i in range(display):
                    debug = self.train_op(sess, batch, count)
                    count += 1

                #On calcule l'avancement et le temps ecoule
                delta_epoch = batch.epoch_count() - delta_epoch
                alepsed_time = time.time() - alepsed_time
                remaining_time = alepsed_time*(epochs-batch.epoch_count())/delta_epoch
                remaining_time = datetime.timedelta(seconds=remaining_time)
                progrssion = batch.epoch_count()/epochs

                #On debug dans la console
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

                if(not debug is None and not debug is {}):
                    line += ", "
                for debug_name in debug:
                    line += debug_name + ": %2.2f"%(debug[debug_name])
                print(line, end='\r')

            print()

    def sess_to_numpy(self, sess):

        for trainable in self.sess_state:
            self.sess_state[trainable] = sess.run(trainable)

    def numpy_to_sess(self):


        self.reset(**self.graph_param)

        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            sess.run(tf.global_variables_initializer())

            for trainable in self.sess_state:
                assign = tf.assign(trainable, self.sess_state[trainable])
                sess.run(assign)

        return sess

    def train_op(self, sess, batch, count):
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
        with tf.variable_scope("conv1_layer"):
            l_l = self.conv(l_l, 8)
            l_l = tf.nn.relu(l_l)
            l_l = htf.pool(l_l, 32)

        with tf.variable_scope("conv2_layer"):
            l_l = self.conv(l_l, 16)
            l_l = tf.nn.relu(l_l)
            l_l = htf.pool(l_l, 16)

        with tf.variable_scope("fcon1_layer"):
            l_l = self.fcon(l_l, 100)
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("fcon2_layer"):
            l_l = self.fcon(l_l, 10)
            return tf.nn.softmax(l_l)


class MnistModel(Model):

    def reset_op(self, **kwargs):

        if("name" in kwargs):
            self.name = kwargs["name"]
        else:
            self.name = "default"

        clas = Classifier('clas', self.is_training)
        clas.set_net()

        self.clas_input = tf.placeholder(tf.float32,
                                         shape=[None, 32, 32, 3],
                                         name="pi_input")

        self.clas_output = clas.output(self.clas_input)

        self.clas_ref_output = tf.placeholder(tf.float32,
                                              shape=[None, 10],
                                              name="pi_ref_output")

        self.clas_loss = htf.celoss(self.clas_output, self.clas_ref_output)
        self.clas_trainer = clas.trainer(self.clas_loss, tf.train.AdamOptimizer(0.0002, 0.5))

    def train_op(self, sess, batch, count):

        batch_x, batch_y_ref = batch.train(100)
        sess.run(self.clas_trainer,feed_dict={self.clas_input:batch_x,
                                                   self.clas_ref_output:batch_y_ref})

        test_x, test_y_ref = batch.test(1000)
        test_y = self.clas_output.eval(session=sess,feed_dict={self.clas_input:test_x})

        return {"acc" : htf.compute_acc(test_y, test_y_ref)}



model = MnistModel()
model.train(batch=Cifar10Batch(), epochs=10)
