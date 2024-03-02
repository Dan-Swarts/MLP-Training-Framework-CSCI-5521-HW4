import keras
from keras import backend as K
print(keras.__version__)

def LReLU(x):
    leakiness = 0.3
    return K.tf.where(K.greater_equal(x, 0), x, leakiness * x)
