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

        self.train_size = len(self.train_file_names)
        self.test_size = len(self.test_file_names)
        self.input_shape = (32, 32, 3)
        self.output_shape = (2)

        self.test_images = np.zeros((self.test_size, 32, 32, 3), dtype="int8")
        self.test_labels = np.zeros((self.test_size, 2), dtype="int8")

        print("Loading test data base...")
        for i in range(len(self.test_file_names)):
            file_name = self.test_file_names[i]
            path = "SkyDataSet/" + file_name + ".jpg"
            image = misc.imread(path)
            height, width, _ = image.shape
            y_top = np.random.randint(height-32)
            x_left = np.random.randint(width-32)
            if(len(image.shape) == 3):
                self.test_images[i]=image[y_top:y_top+32,x_left:x_left+32,:]
            else:
                self.test_images[i,:,:,0]=image[y_top:y_top+32,x_left:x_left+32]
                self.test_images[i,:,:,1]=image[y_top:y_top+32,x_left:x_left+32]
                self.test_images[i,:,:,2]=image[y_top:y_top+32,x_left:x_left+32]
            path = "SkyDataSet/" + file_name + "-skymask.png"
            image = misc.imread(path)
            self.test_labels[i, 0] = image[y_top+16,x_left+16]//255
        self.test_labels[:, 1] = 1 - self.test_labels[:, 0]

        print("Testing size: " + str(self.test_size))
        print("Number of positives: " + str(np.sum(self.test_labels[:, 0])))

        # Now, we make random undeterministic again
        np.random.seed(int(time.time()))

        self.train_images = np.zeros((self.train_size, 32, 32, 3), dtype="int8")
        self.train_labels = np.zeros((self.train_size, 2), dtype="int8")

        print("Calling reload_train for the first time:")
        self.reload_train()

    def reload_train(self):

        np.random.shuffle(self.train_file_names)

        print("Reloading train data base...")
        for i in range(len(self.train_file_names)):
            file_name = self.train_file_names[i]
            path = "SkyDataSet/" + file_name + ".jpg"
            image = misc.imread(path)
            height, width = image.shape[0],image.shape[1]
            y_top = np.random.randint(height-32)
            x_left = np.random.randint(width-32)
            if(len(image.shape) == 3):
                self.train_images[i]=image[y_top:y_top+32,x_left:x_left+32,:]
            else:
                self.train_images[i,:,:,0]=image[y_top:y_top+32,x_left:x_left+32]
                self.train_images[i,:,:,1]=image[y_top:y_top+32,x_left:x_left+32]
                self.train_images[i,:,:,2]=image[y_top:y_top+32,x_left:x_left+32]
            path = "SkyDataSet/" + file_name + "-skymask.png"
            image = misc.imread(path)
            self.train_labels[i, 0] = image[y_top+16,x_left+16]//255
        self.train_labels[:, 1] = 1 - self.train_labels[:, 0]

    def train_op(self, size):

        if(size + (self.count%self.train_size) < self.train_size):

            start = self.count%self.train_size
            end = start + size
            images = self.train_images[start:end]*1.0/255
            labels = self.train_labels[start:end]*1.0

        else:
            self.reload_train()
            images = self.train_images[:size]*1.0/255
            labels = self.train_labels[:size]*1.0

        return images, labels


    def test_op(self, size=0):

        index = np.arange(self.test_size)
        np.random.shuffle(index)
        index = index[:size]

        return self.test_images[index]*1.0/255, self.test_labels[index]*1.0


if(__name__ == "__main__"):

    batch = SkyBatch()

    images, labels = batch.train(size=100)
    print(images.shape)
    print(labels.shape)

    images, labels = batch.test(size=100)
    print(images.shape)
    print(labels.shape)
