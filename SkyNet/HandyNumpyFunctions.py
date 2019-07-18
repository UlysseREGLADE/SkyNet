import numpy as np
import cv2

def compute_acc(y, y_ref):
    return np.mean(np.argmax(y,axis=1)==np.argmax(y_ref,axis=1))

def random_crop(x, scale_factor=1.00, mirror=False):

    b_size = x.shape[0]
    height, width = x.shape[1], x.shape[2]

    if(mirror):

        to_mirror = np.arange(b_size)
        np.random.shuffle(to_mirror)
        x[to_mirror[:b_size//2],:,:,:] = x[to_mirror[:b_size//2],:,::-1,:]

    scale_factor = 1 + np.random.rand()*scale_factor
    resized_height = int(scale_factor*height)
    resized_width = int(scale_factor*width)

    crop_w = np.random.randint(resized_width-width, size=(b_size))
    crop_h = np.random.randint(resized_height-height, size=(b_size))

    ret = np.zeros(x.shape, dtype=x.dtype)

    for i in range(b_size):

        resized_x = cv2.resize(x[i],(resized_width, resized_height))

        ret[i] = resized_x[crop_h[i]:height+crop_y[i],
                           crop_w[i]:width+crop_x[i], :]

    return ret

if(__name__ == "__main__"):

    x = np.zeros((3, 256, 200, 5))

    x = random_crop(x, mirror=True)

    print(x.shape)
