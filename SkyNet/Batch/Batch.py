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
        ret = self.train_op(size)
        size  = min(self.train_size, size)
        self.count += size
        return ret

    def train_op(self, size):
        raise NotImplementedError

    def test(self, size):
        size  = min(self.train_size, size)
        return self.train_op(size)

    def test_op(self, size):
        raise NotImplementedError
