import tensorflow as tf
from SkyNet.Batch import Batch

class MnistBatch(Batch.Batch):

    def load(self):

        from tensorflow.examples.tutorials.mnist import input_data

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.train_size = 60000
        self.test_size = 10000

    def train_op(self, size):

        x_train, y_train = self.mnist.train.next_batch(size)
        x_train = x_train.reshape((size, 28, 28, 1))

        return x_train, y_train

    def test_op(self, size):

        index = np.arange(self.test_size)
        np.random.shuffle(index)
        index = index[:size]

        x_train,y_train=mnist.test.images[index,:],mnist.test.labels[index,:]
        x_train = x_train.reshape((size, 28, 28, 1))

        return x_train, y_train
