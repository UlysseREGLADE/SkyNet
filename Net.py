import tensorflow as tf
import numpy as np
import warnings

class Net(object):

    #Constructeur
    def __init__(self, name, is_training=True):
        #Nom du variable_scope du graph
        self.name = name
        #Ces variables permettent de gerer la batch_norm si besoin
        self.is_training = is_training
        self.is_batch_norm = False
        #Nombre de variables dans le reseau
        self.n_parameters = 0
        #Coefficient pour l'initialisation de xavier
        self.std_coef = 1
        #Initialisation par defaut
        self.default_init = "uniform"

    def output(self, i_input):
        self.scopes_and_variables = []
        self.n_parameters = 0
        with tf.varible_scope(self.name, reuse = tf.AUTO_REUSE):
            l_output = self.net(i_input)
        return l_output

    def net(self):
        #C'est dans cette fonction que doit etre defini le graph de calcule
        raise NotImplementedError

    def set_net(self):
        #C'est dans cette fonction que doit etre defini le graph de calcule
        warnings.warn('def_net not implemented, but called')

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

    def conv(self, x, out_channels, kernel=3, init=None, name="conv"):
        name = self.get_name(name)
        with tf.variable_scope(name):
            #On calcule la variance relative a l'initialisation des poids
            shape = x.get_shape().as_list()
            in_channels = shape[3]
            if(init is None):
                init = self.default_init
            if(init=="normal"):
                stddev = (2/(shape[1]*shape[2]*(in_channels+out_channels)))**0.5
            elif(init=="uniform"):
                stddev = (6/(shape[1]*shape[2]*(in_channels+out_channels)))**0.5
            stddev *= self.std_coef
            #Puis on initialise les variables
            w = self.var([kernel, kernel, in_channels, out_channels],
                           stddev,
                           init,
                           name="weigth")
            b = self.var([out_channels], 0, name="bias")
            #Enfin, on retourne ce qu'il faut
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
            return (conv + b)

    def fcon(self, x, out_size, init=None, name="fcon"):
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
            if(init is None):
                init = self.default_init
            if(init=="normal"):
                stddev = (2/(shape[1]*shape[2]*(in_channels+out_channels)))**0.5
            elif(init=="uniform"):
                stddev = (6/(shape[1]*shape[2]*(in_channels+out_channels)))**0.5
            stddev *= self.std_coef
            #Puis on initialise les variables
            w = self.var([in_size, out_size], stddev, name="weigth")
            b = self.var([out_size], 0, name="bias")
            #Enfin, on retourne ce qu'il faut
            return tf.matmul(x, w) + b

    def var(self, shape, stddev=0, init="const", name="var"):
        #On manage le nom de la variable pour ne pas avoir de pb
        name = self.get_name(name)
        #On calcule l'initialiseur
        if(init=="normal"):
            init=tf.truncated_normal(shape, stddev=stddev)
        elif(init=="uniform"):
            init=tf.random_uniform(shape, stddev=stddev)
        elif(init=="const"):
            init = stddev*np.ones((shape))
        #Puis on la creer
        var = tf.get_variable(name=name, initializer=init)
        #actualisation du nombre de variables dans le graph
        n_param = 1
        for axis in shape:
            n_param *= int(axis)
        self.n_parameters += n_param
        return var

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
