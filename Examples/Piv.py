import tensorflow as tf
import numpy as np

import os
import sys
sys.path.append('../')
from SkyNet.Net import Net
from SkyNet.Model import Model
from SkyNet.Batch.PivBatch import PivBatch
import SkyNet.HandyTensorFunctions as htf

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class Piv(Net):

    def set_net(self, param = {}):
        if("layer" in param):
            self.layer = param["layer"]
        else:
            self.layer = "lrelu"

    def net(self, l_l):
        #Construction du classifieur
        with tf.variable_scope("conv1_layer"):
            l_l = self.conv(l_l, 32, kernel=5)
            l_l = tf.nn.relu(l_l)
            l_l = htf.pool(l_l)

        with tf.variable_scope("conv2_layer"):
            l_l = self.conv(l_l, 32, kernel=5)
            l_l = tf.nn.relu(l_l)
            l_l = htf.pool(l_l)

        with tf.variable_scope("conv2_layer"):
            l_l = self.conv(l_l, 32, kernel=3)
            l_l = tf.nn.relu(l_l)
            l_l = htf.pool(l_l)

        with tf.variable_scope("conv2_layer"):
            l_l = self.conv(l_l, 32, kernel=3)
            l_l = tf.nn.relu(l_l)
            l_l = htf.pool(l_l)

        with tf.variable_scope("fcon2_layer"):
            l_l = self.fcon(l_l, 6)
            return tf.nn.sigmoid(l_l)


class PIVModel(Model):

    def reset_op(self, **kwargs):

        piv = Piv('clas', self.is_training)
        piv.set_net()

        self.norm = np.array([1/4, 1/4, 1/0.05, 1/0.05, 1/0.05, 1/0.05])

        self.input = tf.placeholder(tf.float32,
                                         shape=[None, 48, 48, 2],
                                         name="pi_input")

        self.output = piv.output(self.input)


        self.output_ref = tf.placeholder(tf.float32,
                                              shape=[None, 6],
                                              name="pi_ref_output")
        self.output_ref_norm = (self.output_ref*self.norm+1)/2

        self.loss = htf.ce2Dloss(self.output, self.output_ref_norm)
        self.lr_rate = tf.placeholder(tf.float32, name = "lr_rate")
        self.trainer = piv.trainer(self.loss,
                                         tf.train.AdamOptimizer(self.lr_rate, 0.5))

    def train_op(self, sess, batch, count):

        batch_x, batch_y_ref = batch.train(100)
        lr_rate = 2e-3*(0.5**(count/2000))
        _, batch_loss = sess.run((self.trainer, self.loss),
                              feed_dict={self.input:batch_x,
                                         self.output_ref:batch_y_ref,
                                         self.lr_rate:lr_rate})

        test_x, test_y_ref = batch.test(100)
        test_y, test_loss = sess.run((self.output, self.loss),
                                       feed_dict={self.input:test_x,
                                                  self.output_ref:test_y_ref})
        test_y = (test_y-0.5)*2/self.norm
        u_error = np.mean(np.sum((test_y[:,0:2]-test_y_ref[:,0:2])**2, axis=1))**0.5

        return {"loss_test" : test_loss,
                "loss_train" : batch_loss,
                "u_error": u_error}

    def output(self, i_input):

        with tf.Session(graph=self.graph) as sess:

            #Affichage du nom du graph entrainne
            print("\nCalculating output:")
            print("Graph name: " + str(self.name))

            #Initialisation de la session
            sess.run(tf.global_variables_initializer())

            #On charge la derniere session s'il le faut
            if(os.path.exists(self.name+"/dump.csv")):

                print("Last checkpoint loaded from: " + self.name+"/dump.csv")
                self.saver.restore(sess, self.name+"/save.ckpt")

            return (self.output.eval(session=sess,feed_dict={self.input:i_input})-0.5)*2/self.norm



model = PIVModel(name="piv_sym_model")
batch = PivBatch(path="piv_database")

model.train(batch, epochs=1, display=10)

# x, y_ref = batch.test(100)
# y = model.output(x)
#
# error = np.mean(np.sum((y[:,0:2]-y_ref[:,0:2])**2, axis=1))**0.5
# print(y[0])
# print(y_ref[0])
# print(error)
