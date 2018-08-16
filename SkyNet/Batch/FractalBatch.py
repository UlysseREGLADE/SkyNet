import os
import numpy as np
import tensorflow as tf
from SkyNet.Batch import Batch

class FractalBatch(Batch):

    def __init__(self, parent_batch):

        self.parent_batch = parent_batch()

        self.train_size = self.parent_batch.train_size
        self.test_size = self.parent_batch.test_size
        self.input_shape = self.parent_batch.input_shape
        self.output_shape = self.parent_batch.output_shape + 1

    def modify_input(self, i_input):
        pass

    def train_op(self, size):
        pass

    def test_op(self, size):
        pass
