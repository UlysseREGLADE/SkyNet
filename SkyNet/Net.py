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
        self.default_init = "normal"

    def output(self, i_input):
        self.scopes_and_variables = []
        self.n_parameters = 0
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            l_output = self.net(i_input)
        return l_output

    def net(self):
        #C'est dans cette fonction que doit etre defini le graph de calcule
        raise NotImplementedError

    def set_net(self):
        #C'est dans cette fonction que doit etre defini le graph de calcule
        warnings.warn('def_net not implemented, but called')

    def get_name(self, name):
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
        is_training = self.is_training
        normalized = tf.layers.batch_normalization(x,
                                                   training=is_training,
                                                   name=name)
        return normalized

    def conv(self, x, out_channels, kernel=3, strides=1, stddev=None, init=None, name="conv", padding="SAME", use_bias=True):
        name = self.get_name(name)
        with tf.variable_scope(name):
            #On calcule la variance relative a l'initialisation des poids
            shape = x.get_shape().as_list()
            in_channels = shape[3]

            if(shape[1] is None):
                shape[1] = 1
            if(shape[2] is None):
                shape[2] = 1

            if(init is None):
                init = self.default_init
            if(init=="normal" and stddev is None):
                stddev = (2/(shape[1]*shape[2]*(in_channels+out_channels)))**0.5
            elif(init=="uniform" and stddev is None):
                stddev = (6/(shape[1]*shape[2]*(in_channels+out_channels)))**0.5
            stddev *= self.std_coef
            #Puis on initialise les variables
            w = self.var([kernel, kernel, in_channels, out_channels],
                         param=stddev,
                         init=init,
                         name="weigth")
            #Enfin, on retourne ce qu'il faut
            conv = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)

            if(use_bias):
                b = self.var([out_channels], 0, name="bias")
                return (conv + b)
            else:
                return conv

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
                stddev = (2/(in_size+out_size))**0.5
            elif(init=="uniform"):
                stddev = (6/(in_size+out_size))**0.5
            stddev *= self.std_coef
            #Puis on initialise les variables
            w = self.var([in_size, out_size], param=stddev, init=init, name="weigth")
            b = self.var([out_size], param=0, init=init, name="bias")
            #Enfin, on retourne ce qu'il faut
            return tf.matmul(x, w) + b

    def biLSTM(self, x, num_hidden, pos_timesteps=1, to_output="all"):
        """
        Bidirectionnal LSTM layer. inspired by a code from Aymeric Damien

        Args:
          num_hidden = number of feature you wanna pull from input
          to_output : a string,"all" to return all lstm outputs,
            "last" to return only the final output. May also be an index object.
          timesteps : number of times each lstm cell will be called.
            Must be one dimension of input x. If None, will take the value of the 2nd
            dimension of input x.
        Returns:
          concatenated outputs of the lstm cells. shape (timesteps, 2 * num_hidden)
        """

        timesteps = x.get_shape().as_list()[pos_timesteps]

        x = tf.unstack(x, timesteps, pos_timesteps)

        fw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden)
        bw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

        if to_output == "all" :
            index = slice(None,None)
        elif to_output == "last" :
            index = -1
        else:
            index = to_output

        outputs = tf.stack(outputs, axis=pos_timesteps)

        return outputs[index]

    def var(self, shape, param=0, init="const", name="var"):
        #On manage le nom de la variable pour ne pas avoir de pb
        name = self.get_name(name)
        #On calcule l'initialiseur
        if(init=="normal"):
            init=tf.truncated_normal(shape, stddev=param)
        elif(init=="uniform"):
            init=tf.random_uniform(shape, stddev=param)
        elif(init=="const"):
            init = param*np.ones((shape), dtype=np.float32)
        #Puis on la creer
        var = tf.get_variable(name=name, initializer=init, dtype=tf.float32)
        #actualisation du nombre de variables dans le graph
        n_param = 1
        for axis in shape:
            n_param *= int(axis)
        self.n_parameters += n_param
        return var

    def trainer(self, loss, trainer=tf.train.AdamOptimizer()):
        #On recupere toutes les variables du graph a entrainer
        #Si il n'y a pas de batch normalisation, c'est tout bon
        if(not self.is_batch_norm):
            return trainer.minimize(loss, var_list=self.trainables)
        #Si non, il faut tenir compte des update_ops
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        with tf.control_dependencies(update_ops):
            return trainer.minimize(loss, var_list=self.trainables)

    @property
    def trainables(self):

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=self.name)
