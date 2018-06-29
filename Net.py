import tensorflow as tf

class Net(object):

    #Constructeur
    def __init__(self, name, is_training):
        self.var_list = []
        self.var_index = 0
        self.n_parameters = 0
        self.name = name
        self.is_training = is_training

    def output(self, i_input):
        with tf.name_scope(self.name):
            l_output = self.net(i_input)
        self.var_index = 0
        return l_output

    def normalize(self, x):
        with tf.name_scope("normalize"):
            is_training = self.is_training
            normalized=tf.contrib.layers.batch_norm(x,
                                                    center=True,
                                                    scale=True,
                                                    is_training=is_training)
            return normalized

    def conv(self, x, out_channels, kernel=3, name="conv"):
        with tf.name_scope(name):
            #On calcule la variance relative a l'initialisation des poids
            shape = x.get_shape().as_list()
            in_channels = shape[3]
            l_stddev = (2/(shape[1]*shape[2]*(in_channels+out_channels)))**0.5
            #Puis on initialise les variables
            l_w = self.var([kernel, kernel, in_channels, out_channels],
                           l_stddev,
                           name="weigth")
            l_b = self.var([out_channels], l_stddev, name="bias")
            #Enfin, on retourne ce qu'il faut
            l_conv = tf.nn.conv2d(x, l_w, strides=[1, 1, 1, 1], padding="SAME")
            return (l_conv + l_b)

    def fcon(self, x, out_size, name="fcon", stddev=None):
        with tf.name_scope(name):
            #On commence par gerer la taille du tenseur d'entree
            shape = x.get_shape().as_list()
            in_size = shape[1]
            if(len(shape) > 2):
                for i in range(2, len(shape)):
                    in_size *= shape[i]
                x = tf.reshape(x, [-1, in_size])
            #On calcule la variance relative a l'initialisation des poids
            if(stddev is None):
                stddev = (2/(in_size+out_size))**0.5
            #Puis on initialise les variables
            l_w = self.var([in_size, out_size], stddev, name="weigth")
            l_b = self.var([out_size], name="bias")
            #Enfin, on retourne ce qu'il faut
            return tf.matmul(x, l_w) + l_b

    def var(self, shape, stddev=0, name="var"):
        l_var = None
        if(self.var_index == len(self.var_list)):
            l_var = tf.Variable(tf.truncated_normal(shape,stddev=stddev),
                                name=name)
            self.var_list.append(l_var)
            tf.summary.histogram(name, l_var)
        else:
            l_var = self.var_list[self.var_index]
        self.var_index += 1
        #calcule du nombre de variables
        shape = self.var_list[-1].shape
        n_param = 1
        for axis in shape:
            n_param *= int(axis)
        self.n_parameters += n_param
        return l_var

    def trainer(self, loss, step=1e-3):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(0.0002, 0.5).minimize(loss,
                                                         var_list=self.var_list)
