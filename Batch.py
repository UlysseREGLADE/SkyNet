import tensorflow as tf

class Batch(object):

    def __init__(self):

        self.train_size = None
        self.test_size = None
        self.count = 0

        self.load()

        if(self.train_size is None or self.test_size is None):
            raise AttributeError

    def load(self):
        raise NotImplementedError

    def reset(self):
        self.count = 0

    def epoch_count(self):
        return self.count/self.train_size

    def train(self, size):
        size  = min(self.train_size, size)
        self.count += size
        return self.train_op(size)

    def train_op(self, size=0):
        raise NotImplementedError

    def test(self, size=None):
        size  = min(self.train_size, size)
        if(not size is None):
            return self.train_op(size)
        return self.train_op(size)

    def test_op(self, size=0):
        raise NotImplementedError

class MnistBatch(Batch):

    def load(self):

        from tensorflow.examples.tutorials.mnist import input_data

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.train_size = 60000
        self.test_size = 10000

    def train_op(self, size):

        x_train, y_train = self.mnist.train.next_batch(size)
        x_train = x_train.reshape((size, 28, 28, 1))

        return x_train, y_train

    def test_op(self, size=1000):

        index = np.arange(self.test_size)
        np.random.shuffle(index)
        index = index[:size]

        x_train,y_train=mnist.test.images[index,:],mnist.test.labels[index,:]
        x_train = x_train.reshape((size, 28, 28, 1))

        return x_train, y_train
