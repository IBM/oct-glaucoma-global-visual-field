"""
Network architecture with loss function and optimizer
"""
import keras
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import Input, Model
from keras.layers import (Dense, Dropout, Activation, GlobalAveragePooling3D, 
                          Conv3D, MaxPooling3D, Dropout, SpatialDropout3D)
from keras.layers import concatenate
from keras import regularizers
from nutsml import KerasNetwork

def mse_metric(y_true, y_pred):
    sq_err = K.square(y_true[:,0] - y_pred[:, 0])
    return K.sum(sq_err)

# square error, weighted by uncertainty
# this assumes that we are predicting only one value. if will break if we are predicting more than one thing
# y_true is [[y0, y0],   [y1, y1]...]
# y_pred is [[m0, var0], [m1, var1]...]
def bnn_loss(y_true, y_pred):

    s = y_pred[:,1]
    
    # K.exp() returns rank 1, make it into rank 2 (ie. a column matrix)
    s = K.expand_dims(K.exp(-1.0 * s))

    # K.expand_dims() makes a rank 1 tensor into rank 2
    sq_err = K.expand_dims(K.square(y_true[:,0] - y_pred[:, 0]))

    # this is a column matrix

    # this is for Keras 2.1.3
    # warning: I should test to make sure that on 2.1.3, it does what I hope it does
    dot = K.batch_dot(s, sq_err, [0, 0])

    # this works for Keras 2.2.4
    # dot = K.batch_dot(s, sq_err)

    return K.sum(dot)

# y_pred is [log(sigma^2)....]
# regularize the aleatropic uncertainty
# lam is a weight
def reg_aleatropic(lam=1.0):
    def f(y_pred):
        return lam * K.sum(y_pred)
    return f
