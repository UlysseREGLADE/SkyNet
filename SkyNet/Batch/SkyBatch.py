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

cor = 10

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

        self.train_size = cor*(len(self.train_file_names)//cor)
        self.test_size = len(self.test_file_names)
        self.input_shape = (32, 32, 3)
        self.output_shape = (2)

        self.test_images = np.zeros((self.test_size, 32, 32, 3), dtype="uint8")
        self.test_labels = np.zeros((self.test_size, 2), dtype="uint8")

        print("Loading test data base...")
        for i in range(len(self.test_file_names)):
            file_name = self.test_file_names[i]
            path = "SkyDataSet/" + file_name + ".jpg"
            image = misc.imread(path)
            height, width, _ = image.shape
            y_top = np.random.randint(height-1)
            x_left = np.random.randint(width-1)
            if(len(image.shape) == 3):
                image = np.pad(image, [[15,16], [15,16], [0,0]], 'edge')
                self.test_images[i]=image[y_top:y_top+32,x_left:x_left+32,:]
            else:
                image = np.pad(image, [[15, 16], [15, 16]], 'edge')
                self.test_images[i,:,:,0]=image[y_top:y_top+32,x_left:x_left+32]
                self.test_images[i,:,:,1]=image[y_top:y_top+32,x_left:x_left+32]
                self.test_images[i,:,:,2]=image[y_top:y_top+32,x_left:x_left+32]
            path = "SkyDataSet/" + file_name + "-skymask.png"
            image = misc.imread(path)
            image = np.pad(image, [[15, 16], [15, 16]], 'edge')
            self.test_labels[i, 0] = image[y_top+16,x_left+16]//255
        self.test_labels[:, 1] = 1 - self.test_labels[:, 0]

        print("Testing size: " + str(self.test_size))
        print("Number of positives: " + str(np.sum(self.test_labels[:, 0])))

        # Now, we make random undeterministic again
        np.random.seed(int(time.time()))

        self.train_images = np.zeros((self.train_size, 32, 32, 3), dtype="uint8")
        self.train_labels = np.zeros((self.train_size, 2), dtype="uint8")

        print("Calling reload_train for the first time...")
        self.reload_train()

    def reload_train(self):

        np.random.shuffle(self.train_file_names)

        for i in range(len(self.train_file_names)//cor):

            file_name = self.train_file_names[i]
            path = "SkyDataSet/" + file_name + ".jpg"
            image = misc.imread(path)
            height, width = image.shape[0], image.shape[1]
            path = "SkyDataSet/" + file_name + "-skymask.png"
            lab_image = misc.imread(path)
            lab_image = np.pad(lab_image, [[15, 16], [15, 16]], 'edge')

            for j in range(cor):

                y_top = np.random.randint(height-1)
                x_left = np.random.randint(width-1)

                if(len(image.shape) == 3):

                    if(j==0):
                        image = np.pad(image, [[15,16], [15,16], [0,0]], 'edge')
                    self.train_images[cor*i+j]=image[y_top:y_top+32,x_left:x_left+32,:]

                else:

                    if(j==0):
                        image = np.pad(image, [[15,16], [15,16]], 'edge')
                    self.train_images[cor*i+j,:,:,0]=image[y_top:y_top+32,x_left:x_left+32]
                    self.train_images[cor*i+j,:,:,1]=image[y_top:y_top+32,x_left:x_left+32]
                    self.train_images[cor*i+j,:,:,2]=image[y_top:y_top+32,x_left:x_left+32]

                self.train_labels[cor*i+j, 0] = lab_image[y_top+16,x_left+16]//255

        self.train_labels[:, 1] = 1 - self.train_labels[:, 0]

        self.rd_training = np.arange(self.train_size)
        np.random.shuffle(self.rd_training)

    def train_op(self, size):

        if(size + (self.count%self.train_size) < self.train_size):

            start = self.count%self.train_size
            end = start + size
            images = self.train_images[self.rd_training[start:end]]*1.0/255
            labels = self.train_labels[self.rd_training[start:end]]*1.0

        else:
            self.reload_train()
            images = self.train_images[self.rd_training[:size]]*1.0/255
            labels = self.train_labels[self.rd_training[:size]]*1.0

        return images, labels


    def test_op(self, size=0):

        index = np.arange(self.test_size)
        np.random.shuffle(index)
        index = index[:size]

        return self.test_images[index]*1.0/255, self.test_labels[index]*1.0

    def test_image(self):

        index = np.random.randint(self.test_size)
        file_name = self.test_file_names[index]
        path = "SkyDataSet/" + file_name + ".jpg"
        image = misc.imread(path)

        return image

if(__name__ == "__main__"):

    import matplotlib.pyplot as plt

    batch = SkyBatch()

    images, labels = batch.train(size=10)
    print(images.shape)
    print(labels.shape)
    print(np.min(images), np.max(images))
    print(labels)

    plt.figure()
    plt.imshow(images[0])
    plt.show(False)

    images, labels = batch.test(size=10)
    print(images.shape)
    print(labels.shape)
    print(np.min(images), np.max(images))
    print(labels)

    plt.figure()
    plt.imshow(images[0])
    plt.show()
