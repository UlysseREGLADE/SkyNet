import numpy as np
import sys
sys.path.append('../../')
from SkyNet.Batch.Batch import Batch


class FractalBatch(Batch):

    def load(self, parent_batch, added_crop):


        self.parent_batch = parent_batch()

        self.input_shape = np.array(self.parent_batch.input_shape)
        self.input_shape[[0, 1]] += added_crop
        self.output_shape = self.parent_batch.output_shape + 1

        ratio = self.parent_batch.output_shape / self.output_shape
        self.train_size = int(self.parent_batch.train_size * ratio)
        self.test_size = int(self.parent_batch.test_size * ratio)

        self.added_crop = added_crop

    def modify_batch(self, batch):

        i_input, output = batch
        size = i_input.shape[0]
        height, width = i_input.shape[1], i_input.shape[2]
        resized_input = np.zeros([size] + list(self.input_shape))

        x_shift, y_shift = np.random.randint(self.added_crop + 1,
                                             size=(2, size))
        x_shift = np.reshape(x_shift, (size, 1, 1))
        y_shift = np.reshape(y_shift, (size, 1, 1))

        ar_width, ar_height = np.arange(width), np.arange(height)
        x_coor = np.zeros((height, width))
        x_coor[:] = ar_width
        y_coor = np.zeros((height, width))
        y_coor.T[:] = ar_height

        x_stack, y_stack = np.zeros((2, size, height, width))
        b_stack = np.ones((size, height, width))
        x_stack[:] = x_coor
        y_stack[:] = y_coor
        x_stack = np.array(x_stack+x_shift, dtype=np.int)
        y_stack = np.array(y_stack+y_shift, dtype=np.int)
        b_stack *= np.reshape(np.arange(size), (size, 1, 1))
        b_stack = np.array(b_stack, dtype=np.int)

        resized_input[b_stack, y_stack, x_stack, :] = i_input

        return resized_input, output

    def train_op(self, size):
        size = int(size * self.output_shape / (self.output_shape - 1))
        batch = self.parent_batch.train(size)
        return self.modify_batch(batch)

    def test_op(self, size):
        size = int(size * self.output_shape / (self.output_shape - 1))
        batch = self.parent_batch.train(size)
        return self.modify_batch(batch)


if(__name__ == "__main__"):

    from SkyNet.Batch.MnistBatch import MnistBatch
    import matplotlib.pyplot as plt

    batch = FractalBatch(parent_batch=MnistBatch, added_crop=4)

    batch_input, batch_output = batch.train(10)

    for i in range(10):
        plt.figure()
        plt.imshow(batch_input[i,:,:,0])
        plt.show()
