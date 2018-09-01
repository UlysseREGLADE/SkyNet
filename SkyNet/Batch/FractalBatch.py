import numpy as np
import sys
if(__name__ == "__main__"):
    sys.path.append('../../')
from SkyNet.Batch.Batch import Batch


class FractalBatch(Batch):

    def load(self, parent_batch, added_crop):


        self.parent_batch = parent_batch()

        self.input_shape = np.array(self.parent_batch.input_shape)
        self.input_shape[[0, 1]] += added_crop
        self.output_shape = self.parent_batch.output_shape + 1

        ratio = self.parent_batch.output_shape / self.output_shape
        self.train_size = self.parent_batch.train_size
        self.test_size = self.parent_batch.test_size

        self.added_crop = added_crop

    def get_stack(self, size, width, height):

        ar_width, ar_height = np.arange(width), np.arange(height)
        x_coor = np.zeros((height, width))
        x_coor[:] = ar_width
        y_coor = np.zeros((height, width))
        y_coor.T[:] = ar_height

        x_stack, y_stack = np.zeros((2, size, height, width), dtype=np.int)
        b_stack = np.ones((size, height, width), dtype=np.int)
        x_stack[:] = x_coor
        y_stack[:] = y_coor
        b_stack *= np.reshape(np.arange(size), (size, 1, 1))

        return b_stack, y_stack, x_stack


    def modify_batch(self, batch):

        i_input, i_output = batch
        size = i_input.shape[0]
        height, width = i_input.shape[1], i_input.shape[2]
        depth = i_input.shape[3]

        # Creation of the positives inputs
        positive_input = np.zeros([size] + list(self.input_shape))
        x_shift, y_shift = np.random.randint(self.added_crop + 1,
                                             size=(2, size, 1, 1))

        b_stack, y_stack, x_stack = self.get_stack(size, height, width)

        positive_input[b_stack, y_stack+y_shift, x_stack+x_shift, :] = i_input
        positive_i_output = i_output

        # Creation of the negatives inputs
        back_ground = np.zeros((size,
                                2*height + self.input_shape[0],
                                2*width + self.input_shape[1],
                                depth))
        back_ground[:, :height, :width, :] = i_input
        back_ground[:, :height, -width:, :] = i_input
        back_ground[:, -height:, :width, :] = i_input
        back_ground[:, -height:, -width:, :] = i_input
        x_shift = np.random.randint(2*width - 2, size=(size, 1, 1)) + 1
        y_shift = np.random.randint(2*height - 2, size=(size, 1, 1)) + 1

        b_stack, y_stack, x_stack = self.get_stack(size, self.input_shape[0], self.input_shape[1])

        negative_input = back_ground[b_stack, y_stack+y_shift, x_stack+x_shift, :]

        # Creating of the outputs
        output = np.zeros((2*size, self.output_shape))
        output[:size, :-1] = i_output
        output[size:, -1] = 1

        return np.concatenate([positive_input, negative_input], axis=0), output

    def train_op(self, size):
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

    print(batch_input.shape)

    for i in range(10, 20):
        plt.figure()
        plt.imshow(batch_input[i,:,:,0])
        plt.show()
