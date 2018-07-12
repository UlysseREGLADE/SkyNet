import os
import h5py
import numpy as np
from Batch import Batch

#Le nombre d'images par .h5py
NB_IMG_H5PY = 128000

#Une fonction pour simplement charger ces fichiers
def load_h5py(file_path):

    file = h5py.File(file_path)

    imageA = np.array(file["ImageA"])
    imageC = np.array(file["ImageC"])

    images = np.stack([imageA, imageC], axis = 3)
    details = np.array(file["AllGenDetails"])

    return images, details

#La classe a proprement parler
class PivBatch(Batch):

    def load(self, path = "/run/media/ulysser/Seagate Expansion Drive/MachineLearning/HDF5Data"):

        self.path = path

        if(os.path.exists(self.path)):

            print("Loading PIV database.")

            #On liste les .h5py et on initialise la taille
            self.files = os.listdir(self.path)

            self.separation = int(len(self.files)*0.8)
            self.train_files = self.files[:self.separation]
            self.test_files = self.files[self.separation:]

            self.train_size = len(self.train_files)*NB_IMG_H5PY
            self.test_size = len(self.test_files)*NB_IMG_H5PY

            #On melange
            np.random.shuffle(self.train_files)
            np.random.shuffle(self.test_files)

            #Indice du fichier courant dans la liste melangee
            self.cur_file_index = 0

            #Liste melangee d'indexs au sein d'un fichier
            self.shuffle_indexs = np.arange(NB_IMG_H5PY)
            np.random.shuffle(self.shuffle_indexs)

            #Chargement du premier fichier de la liste melangee
            file_path = self.path+"/"+self.files[self.cur_file_index]
            self.train_images, self.train_details = load_h5py(file_path)

            #Le chargement d'un .h5py etant long, on se perment de precharger le fichier de test
            self.test_count = 0
            file_path = self.path+"/"+self.files[np.random.randint(self.separation, len(self.files))]
            self.test_images, self.test_details = load_h5py(file_path)

            print("PIV database loaded")
            print("train set size: " + str(self.train_size))
            print("test set size: " + str(self.test_size))

    def train_op(self, size):

        #On verifie si on doit charger le fichier suivant
        if(not self.count%NB_IMG_H5PY + size < NB_IMG_H5PY):

            self.cur_file_index += 1

            #On verifie s'il faut remelanger les fichiers
            if(self.cur_file_index > len(self.train_files)):

                np.random.shuffle(self.train_files)
                self.cur_file_index = 0

            #Puis on recharge le fichier
            file_path = self.path+"/"+self.train_files[self.cur_file_index]
            self.train_images, self.train_details = load_h5py(file_path)

            #Et on remelange les indexs aussi
            np.random.shuffle(self.shuffle_indexs)

        #On prend les indexs melanges
        start = self.count%NB_IMG_H5PY
        end = start + size
        indexs = self.shuffle_indexs[start:end]

        #On retourne les images et les proprietes
        return self.train_images[indexs], self.train_details[indexs]


    def test_op(self, size):

        #On choisi au hazard les indexs a tirer
        indexs = np.arange(NB_IMG_H5PY)
        np.random.shuffle(indexs)
        indexs = indexs[:size]

        #On actualise le nombre de testes
        self.test_count += size
        print(self.test_count)

        #Si on a tire trop de fichiers, on change de fichier
        if(not self.test_count < NB_IMG_H5PY):
            print("bonjour")
            self.test_count = 0
            file_path = self.path+"/"+self.files[np.random.randint(self.separation, len(self.files))]
            self.test_images, self.test_details = load_h5py(file_path)

        #On retourne ce qu'il faut
        return self.test_images[indexs], self.test_details[indexs]



if(__name__ == "__main__"):
    batch = PivBatch()

    for i in range(256):
        print(i, batch.test(1000)[0].shape)
