import tensorflow as tf

class Net(object):

    #Constructeur
    def __init__(self, name, is_training=True):
        self.name = name
        self.is_training = is_training
        self.is_batch_norm = False
        self.n_parameters = 0

    def output(self, i_input):
        self.scopes_and_variables = []
        self.n_parameters = 0
        with tf.varible_scope(self.name, reuse = tf.AUTO_REUSE):
            l_output = self.net(i_input)
        return l_output

    def get_name(name):
        index=''
        full_name = tf.get_default_graph().get_name_scope()+"/"+name
        while(full_name+index in self.scopes_and_variables):
            if(index==''):
                index = '_1'
            else:
                index = str(int(index.split('_')[1])+1)
        name += index
        self.scopes_and_variables.append(name)
        return name

    def batch_norm(self, x, name="batch_norm"):
        name = self.get_name(name)
        self.is_batch_norm = True
        with tf.variable_scope(name):
            is_training = self.is_training
            normalized = tf.contrib.layers.batch_norm(x,
                                                      center=True,
                                                      scale=True,
                                                      is_training=is_training)
            return normalized

    def conv(self, x, out_channels, kernel=3, name="conv"):
        name = self.get_name(name)
        with tf.variable_scope(name):
            #On calcule la variance relative a l'initialisation des poids
            shape = x.get_shape().as_list()
            in_channels = shape[3]
            stddev = (2/(shape[1]*shape[2]*(in_channels+out_channels)))**0.5
            init = tf.truncated_normal(shape, stddev=stddev)
            #Puis on initialise les variables
            l_w = self.var([kernel, kernel, in_channels, out_channels],
                           l_stddev,
                           name="weigth")
            l_b = self.var([out_channels], l_stddev, name="bias")
            #Enfin, on retourne ce qu'il faut
            l_conv = tf.nn.conv2d(x, l_w, strides=[1, 1, 1, 1], padding="SAME")
            return (l_conv + l_b)

    def fcon(self, x, out_size, stddev=None, name="fcon"):
        name = self.get_name(name)
        with tf.variable_scope(name):
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
        #On manage le nom de la variable pour ne pas avoir de pb
        name = self.get_name(name)
        #Puis on la creer
        var = tf.get_variable(name=name,
                              initializer=tf.truncated_normal(shape,
                                                              stddev=stddev))
        #calcule du nombre de variables
        shape = var.shape
        n_param = 1
        for axis in shape:
            n_param *= int(axis)
        self.n_parameters += n_param
        return l_var

    def trainer(self, loss, trainer=tf.train.AdamOptimizer()):
        #On recupere toutes les variables du graph a entrainer
        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=self.name)
        #Si il n'y a pas de batch normalisation, c'est tout bon
        if(not self.is_batch_norm):
            return trainer.minimize(loss, var_list=trainables)
        #Si non, il faut tenir compte des update_ops
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        with tf.control_dependencies(update_ops):
            return trainer.minimize(loss, var_list=trainables)
