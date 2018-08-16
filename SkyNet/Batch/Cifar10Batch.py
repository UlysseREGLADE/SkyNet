import numpy as np
import tarfile
import requests
from SkyNet.Batch.Batch import Batch
import os
import pickle

class Cifar10Batch(Batch):

    def load(self):

        #Si CIFAR10 n'est pas telecharge, on le fait
        if(not os.path.exists("cifar-10-python.tar.gz")):

            print("Downloading cifar10 from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
            print("(Can take a few seconds)")

            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            r = requests.get(url, allow_redirects=True)
            open('cifar-10-python.tar.gz', 'wb').write(r.content)

        #Liste des ficheirs a recuperer dans le tar.gz
        file_names = ["cifar-10-batches-py/data_batch_1",
                      "cifar-10-batches-py/data_batch_2",
                      "cifar-10-batches-py/data_batch_3",
                      "cifar-10-batches-py/data_batch_4",
                      "cifar-10-batches-py/data_batch_5",
                      "cifar-10-batches-py/test_batch"]

        #Alloaction des tableux en memoire et init de la taille
        self.train_size = 50000
        self.test_size = 10000

        self.input_shape = (32, 32, 3)
        self.output_shape = (10)

        self.train_images = np.zeros((50000, 32, 32, 3))
        self.train_labels = np.zeros((50000, 10))

        #Puis on lit l'archive a proprement parler
        print("Loading cifar10")

        tar = tarfile.open("cifar-10-python.tar.gz", "r:gz")

        train_count = 0
        for member in tar.getmembers():

            if(member.name in file_names):

                f=tar.extractfile(member)
                data = pickle.load(f, encoding="bytes")

                images = np.array(data[b'data'])
                labels = np.array(data[b'labels'])

                images = np.reshape(images, (10000, 3, 32, 32))
                images = np.transpose(images, (0, 2, 3, 1))
                labels_vect = np.zeros((10000, 10))
                labels_vect[np.arange(10000), labels] = 1
                labels = labels_vect

                if(member.name != "cifar-10-batches-py/test_batch"):

                    self.train_images[10000*train_count:10000*(train_count+1)] = images
                    self.train_labels[10000*train_count:10000*(train_count+1)] = labels

                    train_count += 1

                else:

                    self.test_images = images
                    self.test_labels = labels

        tar.close()

        #Initialisation du random pour le training
        self.rd_training = np.arange(50000)
        np.random.shuffle(self.rd_training)

        #On debug le chargement de cifra
        print("Loading done:")
        print("Training set:")
        print(self.train_images.shape)
        print(self.train_labels.shape)
        print("Test set:")
        print(self.test_images.shape)
        print(self.test_labels.shape)
        print()

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
    batch = Cifar10Batch()
    train = batch.train(100)
    print(train[0].shape, train[1].shape)
    test = batch.test(100)
    print(test[0].shape, test[1].shape)
