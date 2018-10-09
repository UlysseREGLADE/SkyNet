import sys
if(__name__ == "__main__"):
    sys.path.append('../../')
import numpy as np
import tarfile
import requests
from SkyNet.Batch.Batch import Batch
import os
from scipy import misc
import time

class SkyBatch(Batch):

    def load(self):

        # On verifie si le fichier est extrait
        if(not os.path.exists("SkyDataSet")):

            # On verifie que le tar.gz a ete telecharge
            if(not os.path.exists("SkyDataSet.tar.gz")):

                # TODO: Mettre sur git la data-base
                pass
                print("Downloading sky-data-set from: https://www..tar.gz")
                print("(Can take a few seconds)")
                url = ""
                r = requests.get(url, allow_redirects=True)

            # On extrait les fichiers pour avoir un acces plus rappide
            with tarfile.open("SkyDataSet.tar.gz", 'tar:gz') as file:
                file.extractall()

        # Lecture de la liste des fichiers
        with open("SkyDataSet/files-with-sky-20.txt", 'r') as file:
            file_names = file.read().splitlines()
            print("Number of images: " + str(len(file_names)))

        # Get a undeterministic instance of random for partitionning data base
        np.random.seed(167643576)
        np.random.shuffle(file_names)
        test_ratio = 0.2
        pivot = int(0.8*len(file_names))
        self.train_file_names = file_names[:pivot]
        self.test_file_names = file_names[pivot:]

        self.train_size = len(self.train_file_names)*4
        self.test_size = len(self.test_file_names)*4
        self.input_shape = (32, 32, 3)
        self.output_shape = (2)

        self.test_input_images = np.zeros((self.test_size, 64, 64, 3),
                                          dtype="int8")
        self.test_output_images = np.zeros((self.test_size, 64, 64, 1),
                                           dtype="int8")

        for i in range(len(self.test_file_names)):
            file_name = self.test_file_names[i]
            path = "SkyDataSet/" + file_name + ".jpg"
            image = misc.imread(path)
            height, width, _ = image.shape
            y_top = np.random.randint(height-64)
            x_left = np.random.randint(width-64)
            self.test_input_images[i]=image[y_top:y_top+64,x_left:x_left+64,:]

        # Now, we make random undeterministic again
        np.random.seed(int(time.time()))

    def train_op(self, size):

        if(size + (self.count%self.train_size) < self.train_size):

            start = self.count%self.train_size
            end = start + size
            images = self.train_images[self.rd_training[start:end]]
            labels = self.train_labels[self.rd_training[start:end]]

        else:

            self.rd_training = np.arange(50000)
            np.random.shuffle(self.rd_training)
            images = self.train_images[self.rd_training[:size]]
            labels = self.train_labels[self.rd_training[:size]]

        return images, labels


    def test_op(self, size=0):

        index = np.arange(self.test_size)
        np.random.shuffle(index)
        index = index[:size]

        return self.test_images[index], self.test_labels[index]


if(__name__ == "__main__"):

    batch = SkyBatch()
