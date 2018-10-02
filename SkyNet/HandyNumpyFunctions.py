import numpy as np

def compute_acc(y, y_ref):
    return np.mean(np.argmax(y,axis=1)==np.argmax(y_ref,axis=1))
