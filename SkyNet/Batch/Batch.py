class Batch(object):

    def __init__(self, **kwargs):

        self.train_size = None
        self.test_size = None
        self.count = 0

        self.load(**kwargs)

        if(self.train_size is None or self.test_size is None):
            raise ValueError("self.train_size and self.train_size must be defined in load")

    def load(self):
        raise NotImplementedError("load must be implemented")

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
        raise NotImplementedError("train_op must be implemented")

    def test(self, size):
        size  = min(self.train_size, size)
        return self.test_op(size)

    def test_op(self, size):
        raise NotImplementedError("test_op must be implemented")
