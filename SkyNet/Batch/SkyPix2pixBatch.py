import sys
if(__name__ == "__main__"):
    sys.path.append('../../')
from SkyNet.Batch.Batch import Batch
import numpy as np
import os
import time


class SkyPix2pixBatch(Batch):

    def load(self, size = 256):

        name = "SkyDataSet_resized"
        path = name + ".tar.gz"

        if(not os.path.exists(name)):

            import tarfile

            if(not os.path.exists(path)):

                from clint.textui import progress
                import requests
                import clint

                print("Please input the hach to download the database:")
                hach = input()
                url = "https://cloud.mines-paristech.fr/index.php/s/"+hach+"/download"
                name = "SkyDataSet_resized"

                print("Downloading sky-data-set from: " + url)
                print("(Can take a few seconds)")
                r = requests.get(url, allow_redirects=True, stream=True)

                with open(path, 'wb') as f:
                    total_length = int(r.headers.get('content-length'))
                    for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                        if chunk:
                            f.write(chunk)
                            f.flush()

            with tarfile.open(path, 'r:gz') as file:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(file)

        self.images = np.load(name+"/"+name.split("_")[0]+"-%ix%i_input.npy"%(size, size))
        self.masks = np.load(name+"/"+name.split("_")[0]+"-%ix%i_output.npy"%(size, size))

        indexs = np.arange(self.images.shape[0])

        np.random.seed(314159)
        np.random.shuffle(indexs)
        np.random.seed(int(time.time()))

        self.anchor = int(0.90 * self.images.shape[0])

        self.train_indexs = indexs[:self.anchor]

        self.test_indexs = indexs[self.anchor:]

        #Alloaction des tableux en memoire et init de la taille
        self.train_size = self.train_indexs.shape[0]
        self.test_size = self.test_indexs.shape[0]

        self.input_shape = (size, size, 3)
        self.output_shape = (size, size, 1)

        self.count = 0

        #On debug le chargement de cifra
        print("Loading done:")
        print("Training set size = %i"%(len(self.train_indexs)))
        print("Test set size = %i"%(len(self.test_indexs)))
        print()

    def train_op(self, size):

        if(size + (self.count%self.train_size) < self.train_size):

            start = self.count%self.train_size
            end = start + size
            images = self.images[self.train_indexs[start:end]]
            masks = self.masks[self.train_indexs[start:end]]

        else:

            np.random.shuffle(self.train_indexs[:self.anchor])
            images = self.images[self.train_indexs[:size]]
            masks = self.masks[self.train_indexs[:size]]

        images = (images.astype(np.float32)/255)-0.5
        masks = (masks.astype(np.float32)/255)-0.5

        return images, masks


    def test_op(self, size=1):

        index = np.arange(self.test_size)
        np.random.shuffle(index)
        index = index[:size]
        index = self.test_indexs[index]

        images = (self.images[index].astype(np.float32)/255)-0.5
        masks = (self.masks[index].astype(np.float32)/255)-0.5

        return images, masks

    def get_test_by_id(self, ids):

        ids = np.array(ids)

        images = (self.images[ids%self.test_size].astype(np.float32)/255)-0.5
        masks = (self.masks[ids%self.test_size].astype(np.float32)/255)-0.5

        return images, masks

if(__name__ == "__main__"):

    batch = SkyPix2pixBatch()
    train = batch.train(1)
    print(train[0].shape, train[1].shape)
    test = batch.test(1)
    print(test[0].shape, test[1].shape)
