import os
import csv
import time
import shutil
import datetime
import numpy as np
import tensorflow as tf

import SkyNet.Net as cn
import SkyNet.HandyTensorFunctions as htf
from SkyNet.Batch.Cifar10Batch import Cifar10Batch

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

            spam_writer.writerow([col for col in sorted(i_dict)])

            data_length = len(next(iter(i_dict.values())))
            for i in range(data_length):
                spam_writer.writerow([i_dict[col][i] for col in sorted(i_dict)])

    else:

        #Sinon, on l'append
        with open(dir+"/"+name, "a") as csv_file:

            spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")

            data_length = len(next(iter(i_dict.values())))
            for i in range(data_length):
                spam_writer.writerow([i_dict[col][i] for col in sorted(i_dict)])

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
        print(ret)
        return ret

    #Sinon, on retourne une erreur
    raise OSError


class Evaluator(object):

    """
    This class serve as a context manager that internaly opens and close
    sessions so that the user has not to deal with tensorflow when it commes
    to using a trained model to make a prediction.

    Example:
        with model.default_evaluator as eval:
            output = eval.compute(input)

    Properties:
        model: a pointer to the parent model
        input: a list of the placeholder to fill to get the output
        output: the output node in the model's graph
    """

    def __init__(self, model, input_list, output):

        """
        Set the clss properties to the given parameters

        Args:
            model: a pointer to the parent model
            input: a list of the placeholder to fill to get the output
            output: the output node in the model's graph
        """

        self.model = model
        self.output = output
        self.input_list = input_list

    def __enter__(self):

        """
        Open a tensorflow session that will be used to compute the output.
        Then load the model from the last checkpoint.

        Return:
            a pointer to it self
        """

        self.sess = tf.Session(graph = self.model.graph)

        #Initialisation de la session

        with self.model.graph.as_default():
            saver = tf.train.Saver()

        #On charge la derniere session
        if(os.path.exists(self.model.name+"/dump.csv")):

            print("Last checkpoint loaded from: " + self.model.name+"/dump.csv")
            saver.restore(self.sess, self.model.name+"/save.ckpt")

        return self

    def __exit__(self, *args, **kwargs):

        """
        Close the tensorflow session to free the CPU or GPU resorce.
        """

        self.sess.close()

    def compute(*i_input):

        """
        Compute the output for the given input.

        Args:
            The number of arguments must match len(self.input_list), and they must
            be given in the same order.

        Return:
            The output of the model evaluated on i_input
        """

        self = i_input[0]

        if(len(self.input_list) != len(i_input)-1):
            raise IndexError("The number of inputs of compute must match the len of input_list")

        feed_dict = {}
        for i in range(len(self.input_list)):
            feed_dict[self.input_list[i]] = i_input[i+1]

        return self.output.eval(session=self.sess,
                                feed_dict=feed_dict)

class Model(object):

    def __init__(self, **kwargs):

        self.reset(**kwargs)


    def reset(self, **kwargs):

        #On reset le graph et ses parametres
        self.graph = tf.Graph()
        self.graph_param = kwargs

        #Gestion du nom du graph (ie sauvegarde)
        self.name = "model"
        if("name" in kwargs):
            self.name = kwargs["name"]

        #Initialisation du graph de calcule
        with self.graph.as_default():

            #Appelle de la fonction reset
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.reset_op(**kwargs)

            #On se donne un sauver
            self.saver = tf.train.Saver()

        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def train(self, batch, epochs, display=100, save=100, overwrite=False):

        """
        When called this function will start the training of the model, and save
        it in a folder named with self.name.

        Args:
            batch: (Batch) the batch generator for the training session
            epochs: (int) The number of time the the training session goes through the
                    entier database.
            display: (int) The refresh rate of the console output
            save: (int) The saving rate of the model
            overwrite: (bool) if the folder named self.name already exists,
                       do we overwrite it or not.
        """

        with tf.Session(graph=self.graph) as sess:

            #Affichage du nom du graph entrainne
            print("\nStarting training:")
            print("Graph name: " + str(self.name))

            #Initialisation de la session
            sess.run(tf.global_variables_initializer())
            batch.reset()

            #On reset le compte
            count = 0

            #On supprime le dossier de sauvegarde s'il faut
            if(overwrite and os.path.exists(self.name)):
                shutil.rmtree(self.name)

            if(not overwrite and os.path.exists(self.name+"/dump.csv")):
            #On charge la derniere session s'il le faut

                print("Last checkpoint loaded from: " + self.name+"/dump.csv")

                last_dump = load_last_dump(self.name+"/"+"dump.csv")
                count = int(float(last_dump["count"]))

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
                    debug = self.train_op(sess, batch, count)
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

    def evaluator(self, input_list, output):

        """
        Allows the user to build an evaluator wich depends on custome input,
        and that can have an output which is not the default one.

        Args:
            input: a list of the placeholder to fill to get the output
            output: the output node in the model's graph

        Return:
            An evaluator wich points to the current model
        """

        return Evaluator(self, input_list, output)

    def default_evaluator(self):

        """
        Allows the user to easily build an evaluator for the current model.
        To work, self.output and self.input_list must be defined within reset_op.

        Return:
            An evaluator wich points to the current model
        """

        return Evaluator(self, self.input_list, self.output)

    def train_op(self, sess, batch, count):

        raise NotImplementedError

    def reset_op(self, **kwargs):

        raise NotImplementedError
