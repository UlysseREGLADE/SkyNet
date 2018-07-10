import os
import csv
import time
import shutil
import datetime
import numpy as np
import tensorflow as tf

import Net as cn
import HandyTensorFunctions as htf
from Batch.Cifar10Batch import Cifar10Batch

"""
For my ally is the Force, and a powerful ally it is. Life creates it, makes it
grow. Its energy surrounds us and binds us. Luminous beings are we, not this
crude matter. You must feel the Force around you; here, between you, me, the
tree, the rock, everywhere, yes. Even between the land and the ship.
"""

def dump_csv(dir, name, i_dict):

    #Si le dossier de sauvegarde n'est pas la
    if(not os.path.exists(dir)):

        #On le cree
        os.mkdir(dir)

    #Maintenant, on gere le cas ou le fichier n'est pas la
    if(not os.path.exists(dir+"/"+name)):

        #Si le fichier n'existe pas, on le cree
        with open(dir+"/"+name, "w") as csv_file:

            spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")

            spam_writer.writerow([col for col in i_dict])

            data_length = len(next(iter(i_dict.values())))
            for i in range(data_length):
                spam_writer.writerow([i_dict[col][i] for col in i_dict])

    else:

        #Sinon, on l'append
        with open(dir+"/"+name, "a") as csv_file:

            spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")

            data_length = len(next(iter(i_dict.values())))
            for i in range(data_length):
                spam_writer.writerow([i_dict[col][i] for col in i_dict])

#TODO: Read the last line of a .csv

def load_last_dump(path):

    #Lecture du fichier

    #On initialise les variables a None
    fisrt_ligne, last_ligne = None, None

    #On lit la premiere ligne
    with open(path, "r") as csv_file:

        spam_reader = csv.reader(csv_file, delimiter=";", lineterminator="\n")
        for row in spam_reader:
            fisrt_ligne = row
            break

    #Ainsi que la derniere
    with open(path, "r") as csv_file:

        spam_reader = csv.reader(csv_file, delimiter=";", lineterminator="\n")
        for row in reversed(list(spam_reader)):
            last_ligne = row
            break

    #Maintenant, si la lecture s'est bien passee
    if(not fisrt_ligne is None and not last_ligne is None):
        ret = {}
        for i in range(len(fisrt_ligne)):
            ret[fisrt_ligne[i]] = last_ligne[i]

        return ret

    #Sinon, on retourne une erreur
    raise OSError


class Model(object):

    def __init__(self, **kwargs):

        self.reset(**kwargs)


    def reset(self, **kwargs):

        #On reset le graph et ses parametres
        self.graph = tf.Graph()
        self.graph_param = kwargs

        #Gestion du nom du graph (ie sauvegarde)
        self.name = "default_model"
        if("name" in kwargs):
            self.name = kwargs["name"]

        #Gestion de la restauration du graph
        self.restore = True
        if("restore" in kwargs):
            self.restore = kwargs["restore"]

        #Initialisation du graph de calcule
        with self.graph.as_default():

            #Appelle de la fonction reset
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.reset_op(**kwargs)

            #On se donne un sauver
            self.saver = tf.train.Saver()

        #On supprime le dossier de sauvegarde s'il faut
        if(not self.restore and os.path.exists(self.name)):
            shutil.rmtree(self.name)

        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def train(self, batch, epochs, batch_size=100, display=100, save=100):

        with tf.Session(graph=self.graph) as sess:

            #Affichage du nom du graph entrainne
            print("\nStarting training:")
            print("Graph name: " + str(self.name))

            #Initialisation de la session
            sess.run(tf.global_variables_initializer())
            batch.reset()

            #On reset le compte
            count = 0

            #On charge la derniere session s'il le faut
            if(self.restore and os.path.exists(self.name+"/dump.csv")):

                print("Last checkpoint loaded from: " + self.name+"/dump.csv")

                last_dump = load_last_dump(self.name+"/"+"dump.csv")
                count = float(last_dump["count"])

                self.saver.restore(sess, self.name+"/save.ckpt")

            #Boucle d'entrainnement
            to_dump = {}
            while(batch.epoch_count()<epochs):

                #Initialisation des varaibles pour calculer le temps
                alepsed_time = time.time()
                delta_epoch = batch.epoch_count()

                #On effectue le trainning
                for i in range(display):

                    #Entrainement
                    debug = self.train_op(sess, batch)
                    debug["count"] = count

                    #Gestion des donnes a dumper dans le .csv
                    if(to_dump == {}):
                        for data in debug:
                            to_dump[data] = [debug[data]]
                    else:
                        for data in debug:
                            to_dump[data].append(debug[data])

                    #On sauve s'il le faut
                    if(count%save == 0):

                        #On sauve l'etat de la session
                        self.saver.save(sess, self.name+"/save.ckpt")

                        #Et on sauve le csv
                        if(to_dump != {}):
                            dump_csv(self.name, "dump.csv", to_dump)
                            to_dump = {}

                    #Actualisation du compteur d'entrainement
                    count += 1

                #On calcule l'avancement et le temps ecoule
                delta_epoch = batch.epoch_count() - delta_epoch
                alepsed_time = time.time() - alepsed_time
                remaining_time = alepsed_time*(epochs-batch.epoch_count())/delta_epoch
                remaining_time = datetime.timedelta(seconds=remaining_time)
                progrssion = batch.epoch_count()/epochs

                #On debug dans la console

                #Barre de chargement
                line = "["
                bar_size = 20
                for i in range(bar_size):
                    if(i<int(bar_size*progrssion)):
                        line += "#"
                    else:
                        line += " "
                line += "] %2.1f"%(100*progrssion) + "% "

                #Affichage du temps
                hours, rem = divmod(remaining_time.seconds, 3600)
                minutes, seconds = divmod(rem, 60)
                line += "%2d day(s), %2dh%2dm%2ds"%(remaining_time.days,
                                                    hours,
                                                    minutes,
                                                    seconds)

                #Affichage des donnees specifiques au model
                if(not debug is None and not debug is {}):
                    for debug_name in debug:
                        line += ", " + debug_name + ": %.2f"%(debug[debug_name])
                print(line, end='\r')

            print()

    def train_op(self, sess, batch):
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
            l_l = self.conv(l_l, 64, kernel=5)
            l_l = tf.nn.relu(l_l)
            l_l = htf.pool(l_l)
            l_l = tf.nn.lrn(l_l, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')

        with tf.variable_scope("conv2_layer"):
            l_l = self.conv(l_l, 64, kernel=5)
            l_l = tf.nn.relu(l_l)
            l_l = htf.pool(l_l)
            l_l = tf.nn.lrn(l_l, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')

        with tf.variable_scope("fcon1_layer"):
            l_l = self.fcon(l_l, 256)
            l_l = tf.nn.relu(l_l)

        with tf.variable_scope("fcon2_layer"):
            l_l = self.fcon(l_l, 10)
            return tf.nn.softmax(l_l)


class MnistModel(Model):

    def reset_op(self, **kwargs):

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
        self.clas_trainer = clas.trainer(self.clas_loss,
                                         tf.train.AdamOptimizer(0.0002, 0.5))

    def train_op(self, sess, batch):

        batch_x, batch_y_ref = batch.train(100)
        _, batch_y = sess.run((self.clas_trainer, self.clas_output),
                              feed_dict={self.clas_input:batch_x,
                                         self.clas_ref_output:batch_y_ref})

        test_x, test_y_ref = batch.test(1000)
        test_y = self.clas_output.eval(session=sess,
                                       feed_dict={self.clas_input:test_x})

        return {"acc_test" : htf.compute_acc(test_y, test_y_ref),
                "acc_train" : htf.compute_acc(batch_y, batch_y_ref)}



model = MnistModel()
model.train(batch=Cifar10Batch(), epochs=10, display=10, save=100)
