import tensorflow as tf

import SkyNet.HandyTensorFunctions as htf

class Net(object):

    #Constructeur
    def __init__(self, name):
        self.var_list = []
        self.var_index = 0
        self.name = name

    def output(self, i_input):
        with tf.name_scope(self.name):
            l_output = self.net(i_input)
        self.var_index = 0
        return l_output

    def normalize(self, x, in_channels):
        with tf.name_scope("normalize"):
            l_mean, l_var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
            #l_offset = self.var([in_channels], 0, name="offset")
            #l_scale = self.var([in_channels], 1, name="scale")
            l_normalized = tf.nn.batch_normalization(x, l_mean, l_var, 0, 1, 1e-5)
            return l_normalized

    def conv(self, x, in_channels, out_channels, kernel=3, name="conv"):
        with tf.name_scope(name):
            l_x = self.normalize(x, in_channels)
            l_stddev = (2/(int(x.shape[1])*int(x.shape[2])*(in_channels+out_channels)))**0.5
            l_w = self.var([kernel, kernel, in_channels, out_channels], l_stddev, name="weigth")
            l_b = self.var([out_channels], l_stddev, name="bias")
            l_conv = tf.nn.conv2d(l_x, l_w, strides=[1, 1, 1, 1], padding="SAME")
            return (l_conv + l_b)

    def fcon(self, x, in_size, out_size, bias=True, name="fcon"):
        with tf.name_scope(name):
            l_x = tf.reshape(x, [-1, in_size])
            l_stddev = (2/(in_size+out_size))**0.5
            l_w = self.var([in_size, out_size], l_stddev, name="weigth")
            l_b = self.var([out_size], l_stddev, name="bias")
            return tf.matmul(l_x, l_w) + l_b

    def var(self, shape, stddev=0, name="var"):
        l_var = None
        if(self.var_index == len(self.var_list)):
            self.var_list.append(tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name))
            l_var = self.var_list[-1]
            tf.summary.histogram(name, l_var)
        else:
            l_var = self.var_list[self.var_index]
        self.var_index += 1
        return l_var

    def trainer(self, loss, step=None):
        if(step == None):
            return tf.train.AdamOptimizer().minimize(loss, var_list = self.var_list)
        else:
            return tf.train.GradientDescentOptimizer(step).minimize(loss, var_list = self.var_list)
