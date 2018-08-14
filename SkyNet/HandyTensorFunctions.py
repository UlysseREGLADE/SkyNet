import tensorflow as tf
import numpy as np

#Activation functions

def lrelu(x, slope=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, slope*x)

def outrelu(x, name="outrelu"):
    with tf.variable_scope(name):
        return tf.minimum(tf.maximum(x, -0.5), 0.5)+0.5

#Loss functions

def celoss(y, y_ref, name="celoss"):
    with tf.variable_scope(name):
        y = tf.clip_by_value(y, 1e-10, 1)
        return tf.reduce_mean(-y_ref*tf.log(y))

def l1loss(y, y_ref, name="l1loss"):
    with tf.variable_scope(name):
        return tf.reduce_mean(tf.abs(tf.subtract(y, y_ref)))

def l2loss(y, y_ref, name="l2loss"):
    with tf.variable_scope(name):
        return tf.reduce_mean((y-y_ref)**2)

def ce2Dloss(y, y_ref, name="ce2Dloss"):
    with tf.variable_scope(name):
        y = tf.clip_by_value(y, 1e-10, 1-(1e-10))
        return tf.reduce_mean(-y_ref*tf.log(y) - (1-y_ref)*tf.log(1-y))

def compute_acc(y, y_ref):
    return np.mean(np.argmax(y,axis=1)==np.argmax(y_ref,axis=1))

#Pooling functions

def mask(rep, tile, one):
    l_mask = np.zeros((1, tile, tile, 1), dtype=np.float32)
    l_mask[0, one[0], one[1], 0] = 1
    l_mask = np.tile(l_mask, (1, rep, rep, 1))
    l_mask = tf.constant(l_mask, name=str(tile)+"x"+str(tile)+"_"+str(one[0])+"-"+str(one[1]))
    return l_mask

def unpool(x, size, mask_size=None, i_mask=None, strides=2, name="unpool"):
    with tf.variable_scope(name):
        #If the mask is not given, creats it
        if(i_mask == None):
            if(mask_size==None):
                mask_size=size*strides
            i_mask = mask(size, strides, [strides//2, strides//2])
            if(strides*size != mask_size):
                l_pad = strides*size - mask_size
                if(l_pad%2 != 0):
                    i_mask = i_mask[:, l_pad//2+1:strides*size-l_pad//2, l_pad//2+1:strides*size-l_pad//2, :]
                else:
                    i_mask = i_mask[:, l_pad//2:strides*size-l_pad//2, l_pad//2:strides*size-l_pad//2, :]
        #Resizing of the input
        l_y = tf.image.resize_images(x, [tf.shape(x)[1]*strides, tf.shape(x)[2]*strides], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #If padding is needed, adjusts the input to the mask
        if(strides*size != mask_size):
            l_pad = strides*size - mask_size
            if(l_pad%2 != 0):
                l_y = l_y[:, l_pad//2+1:strides*size-l_pad//2, l_pad//2+1:strides*size-l_pad//2, :]
            else:
                l_y = l_y[:, l_pad//2:strides*size-l_pad//2, l_pad//2:strides*size-l_pad//2, :]
        #return the unpooled result
        return l_y*i_mask

def pool(x, i_mask=False, strides=2, name="pool"):
    with tf.variable_scope(name):

        #Extracting the size of the input
        shape = x.get_shape().as_list()
        size = shape[1]

        #We adjust image size to the strides
        l_pad1, l_pad2 = 0, 0
        if(size%strides != 0):
            l_pad = size%strides
            size += l_pad
            l_pad1, l_pad2 = l_pad//2, l_pad//2
            if(l_pad%2 != 0):
                l_pad1 += 1
            x = tf.pad(x, [[0,0],[l_pad1,l_pad2],[l_pad1,l_pad2],[0,0]], mode="CONSTANT")
        #Then the input is pooled
        l_y = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], padding="VALID")
        #if necessary, the complicated part: calculation of the mask
        if(i_mask):
            l_unpools = []
            l_pools = []
            l_placed = tf.zeros([1, size//strides, size//strides, 1], tf.bool)
            l_mask = tf.zeros([1, size, size, 1])
            for i in range(strides):
                for j in range(strides):
                    l_unpools.append(mask(size//strides, strides, [i, j]))
                    l_unpools[-1] = l_unpools[-1]
                    l_pools.append(tf.nn.max_pool(l_unpools[-1]*x, [1, strides, strides, 1], [1, strides, strides, 1], padding="VALID"))
                    l_pools[-1] = tf.equal(l_pools[-1], l_y)
            for i in range(strides**2):
                l_maskCourant = tf.logical_and(tf.logical_not(l_placed), l_pools[i])
                l_placed = tf.logical_or(l_placed, l_pools[i])
                l_maskCourant = tf.cast(l_maskCourant, tf.float32)
                l_maskCourant = tf.image.resize_images(l_maskCourant, [size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                l_mask = l_mask + l_maskCourant*l_unpools[i]
            #returns the output and the calculated mask
            return l_y, l_mask[:, l_pad1:size-l_pad2, l_pad1:size-l_pad2, :]
        else:
            return l_y
