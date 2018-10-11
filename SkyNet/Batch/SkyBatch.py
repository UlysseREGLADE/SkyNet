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

COR = 10
SIZE = 32

def format_image(image):

    height, width = image.shape[0], image.shape[1]

    reshaped_image = np.zeros((height+SIZE-1+SIZE%2, width+SIZE-1+SIZE%2, 5))

    grad_x = np.zeros((height+SIZE-1+SIZE%2, width+SIZE-1+SIZE%2))
    grad_x[:] = np.linspace(0, 1, width+SIZE-1+SIZE%2)
    grad_y = np.zeros((height+SIZE-1+SIZE%2, width+SIZE-1+SIZE%2))
    grad_y.T[:] = np.linspace(0, 1, height+SIZE-1+SIZE%2)

    if(len(image.shape)==3):
        reshaped_image[:,:,0:3] = np.pad(image, [[SIZE//2-1+SIZE%2, SIZE//2], [SIZE//2-1+SIZE%2, SIZE//2], [0,0]], 'edge')/255
    else:
        reshaped_image[:,:,0] = np.pad(image, [[SIZE//2-1+SIZE%2, SIZE//2], [SIZE//2-1+SIZE%2, SIZE//2]], 'edge')/255
        reshaped_image[:,:,1] = reshaped_image[:,:,0]
        reshaped_image[:,:,2] = reshaped_image[:,:,0]

    reshaped_image[:,:,3] = grad_x
    reshaped_image[:,:,4] = grad_y

    return reshaped_image

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
        np.random.seed(int(time.time()))

        test_ratio = 0.2
        pivot = int(0.8*len(file_names))
        self.train_file_names = file_names[:pivot]
        self.test_file_names = file_names[pivot:]

        self.train_size = COR*(len(self.train_file_names)//COR)
        self.test_size = len(self.test_file_names)
        self.input_shape = (SIZE, SIZE, 5)
        self.output_shape = (2)

        self.test_images = np.zeros((self.test_size, SIZE, SIZE, 5))
        self.test_labels = np.zeros((self.test_size, 2), dtype="uint8")

        self.train_images = np.zeros((self.train_size, SIZE, SIZE, 5))
        self.train_labels = np.zeros((self.train_size, 2), dtype="uint8")

    def load_test(self):

        for i in range(len(self.test_file_names)):

            file_name = self.test_file_names[i]
            path = "SkyDataSet/" + file_name + ".jpg"
            image = misc.imread(path)
            height, width, _ = image.shape
            y_top = np.random.randint(height-1)
            x_left = np.random.randint(width-1)

            formated_image = format_image(image)

            self.test_images[i] = formated_image[y_top:y_top+SIZE,x_left:x_left+SIZE,:]

            path = "SkyDataSet/" + file_name + "-skymask.png"
            image = misc.imread(path)
            image = np.pad(image, [[SIZE//2-1+SIZE%2, SIZE//2], [SIZE//2-1+SIZE%2, SIZE//2]], 'edge')
            self.test_labels[i, 0] = image[y_top+SIZE//2,x_left+SIZE//2]//255

        self.test_labels[:, 1] = 1 - self.test_labels[:, 0]

        print("Testing size: " + str(self.test_size))
        print("Number of positives: " + str(np.sum(self.test_labels[:, 0])))

    def reload_train(self):

        np.random.shuffle(self.train_file_names)

        for i in range(len(self.train_file_names)//COR):

            file_name = self.train_file_names[i]
            path = "SkyDataSet/" + file_name + ".jpg"
            image = misc.imread(path)
            height, width = image.shape[0], image.shape[1]
            path = "SkyDataSet/" + file_name + "-skymask.png"
            lab_image = misc.imread(path)
            lab_image = np.pad(lab_image, [[SIZE//2-1+SIZE%2, SIZE//2], [SIZE//2-1+SIZE%2, SIZE//2]], 'edge')

            formated_image = format_image(image)

            for j in range(COR):

                y_top = np.random.randint(height-1)
                x_left = np.random.randint(width-1)

                self.train_images[COR*i+j] = formated_image[y_top:y_top+SIZE,x_left:x_left+SIZE,:]

                self.train_labels[COR*i+j, 0] = lab_image[y_top+SIZE//2,x_left+SIZE//2]//255

        self.train_labels[:, 1] = 1 - self.train_labels[:, 0]

        self.rd_training = np.arange(self.train_size)
        np.random.shuffle(self.rd_training)

    def train_op(self, size):

        if(not hasattr(self, 'train_flag')):
            self.train_flag = True
            print("Calling reload_train for the first time...")
            self.reload_train()

        if(size + (self.count%self.train_size) < self.train_size):

            start = self.count%self.train_size
            end = start + size
            images = self.train_images[self.rd_training[start:end]]
            labels = self.train_labels[self.rd_training[start:end]]*1.0

        else:
            self.reload_train()
            images = self.train_images[self.rd_training[:size]]
            labels = self.train_labels[self.rd_training[:size]]*1.0

        return images, labels


    def test_op(self, size=0):

        if(not hasattr(self, 'test_flag')):
            self.train_flag = True
            print("Loading test data...")
            self.load_test()

        index = np.arange(self.test_size)
        np.random.shuffle(index)
        index = index[:size]

        return self.test_images[index], self.test_labels[index]*1.0

    def test_image(self):

        if(not hasattr(self, 'test_image_index')):
            self.test_image_index = 0
        else:
            self.test_image_index = (self.test_image_index+1)%self.test_size

        file_name = self.test_file_names[self.test_image_index]
        path = "SkyDataSet/" + file_name + ".jpg"
        image = misc.imread(path)
        path = "SkyDataSet/" + file_name + "-skymask.png"
        lab_image = misc.imread(path)

        return image, lab_image

if(__name__ == "__main__"):

    import matplotlib.pyplot as plt

    batch = SkyBatch()

    images, labels = batch.train(size=10)
    print(images.shape)
    print(labels.shape)
    print(np.min(images), np.max(images))
    print(labels)

    plt.figure()
    plt.imshow(images[0,:,:,0:3])
    plt.show(False)

    plt.figure()
    plt.imshow(images[0,:,:,3])
    plt.show(False)

    plt.figure()
    plt.imshow(images[0,:,:,4])
    plt.show(False)

    images, labels = batch.test(size=10)
    print(images.shape)
    print(labels.shape)
    print(np.min(images), np.max(images))
    print(labels)

    plt.figure()
    plt.imshow(images[0,:,:,0:3])
    plt.show()
