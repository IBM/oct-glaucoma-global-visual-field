from keras import backend as K
from keras.regularizers import Regularizer
from keras.layers import Layer

# calculate the covariance matrix of a batch
def cov(batch):
    m = K.expand_dims(K.mean(batch, axis=0), axis=0)
    n_row = K.eval(K.shape(batch))[0]
    m = K.repeat_elements(m, n_row, axis=0)
    batch2 = batch - m # remove the mean
    return K.dot(K.transpose(batch2), batch)/(n_row-1)

def forbenius_norm(m):
    m2 = K.expand_dims(K.flatten(m), axis=0)
    return K.dot(m2, K.transpose(m2))[0][0]

def diag_norm(m):
    d = K.expand_dims(tf.diag_part(m), axis=0)
    return K.dot(d, K.transpose(d))[0][0]

def decorrelation_reg(batch):
    corr = cov(batch)
        
    term1 = forbenius_norm(corr)
    term2 = diag_norm(corr)
    return (term1 - term2)

class DecorrelationRegularizer(Regularizer):
    def __init__(self, l=0.):
        self.l = K.cast_to_floatx(l)

    def __call__(self, x):
        return self.l * 0.0

    def get_config(self):
        return {'l': float(self.l)}

class CorrelationRegularization(Layer):

    def __init__(self, l=0., **kwargs):
        super(CorrelationRegularization, self).__init__(**kwargs)
        self.supports_masking = True
        self.l = l
        self.activity_regularizer = DecorrelationRegularizer(l=l)

    def get_config(self):
        config = {'l': self.l}
        base_config = super(CorrelationRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
