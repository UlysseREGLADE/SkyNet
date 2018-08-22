import os
import numpy as np
import tensorflow as tf
from SkyNet.Batch import Batch

class FractalBatch(Batch):

    def __init__(self, parent_batch, added_crop):

        self.parent_batch = parent_batch()

        self.input_shape = self.parent_batch.input_shape
        self.input_shape[[0,1]] += added_crop
        self.output_shape = self.parent_batch.output_shape + 1

        ratio = self.parent_batch.output_shape/self.output_shape
        self.train_size = int(self.parent_batch.train_size*ratio)
        self.test_size = int(self.parent_batch.test_size*ratio)

    def modify_batch(self, batch):
        x, y = batch


    def train_op(self, size):
        size = int(size*self.output_shape/(self.output_shape-1))
        batch = self.parent_batch.train(size)
        return self.modify_batch(batch)

    def test_op(self, size):
        size = int(size*self.output_shape/(self.output_shape-1))
        batch = self.parent_batch.train(size)
        return self.modify_batch(batch)
