import tensorflow as tf
from enum import Enum
from . import utilities as tfhu

class init():
#     def zeros_variable(shape):
#         return tf.Variable(tf.zeros(shape))
        
    def weight_variable(shape):
        print("weight_variable",shape)
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

#     def conv2d(x, W):
#         return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#     def max_pool_2x2(x):
#         return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
#             strides=[1, 2, 2, 1], padding='SAME')
            
#     def max_pool_4x4(x):
#         return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], 
#             strides=[1, 4, 4, 1], padding='SAME')

# class initType(Enum):
#     zero = 1
    
class Layer(object):
    """
    Abstract Class
    Layer with Inputs and Outputs
    """
    
    def getInputShape(self, *args):
        """Overloaded for Layer.validateInput()"""
        raise NotImplementedError
        
    def getInputDtype(self, *args):
        """Overloaded for Layer.validateInput()"""
        raise NotImplementedError
            
    def connect(self, *args):
        raise NotImplementedError
        
    def validateTFInput(name, tfLayerInput, shape, dtype):
        tfhu.validate(name, "Tensor shape", shape, tfLayerInput._shape_as_list())
        tfhu.validate(name, "Tensor dtype", dtype, tfLayerInput.dtype)
        
    def validateInput(self, tfLayerInput):
        """Check if the Layer's shape and type match with ths input's"""
        Layer.validateTFInput(self.name, tfLayerInput, shape=self.getInputShape(), dtype=self.getInputDtype())

class ValidationLayer(Layer):
    def __init__(self, shape=None, dtype=None, name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        
    def getInputShape(self, *args):
        return self.shape
        
    def getInputDtype(self, *args):
        return self.dtype
        
    def connect(self, prevLayerResult):
        self.validateInput(prevLayerResult)
        return prevLayerResult
 
class _OpLayer(Layer):
    """
    Abstract Class
    Layer for doing Native TensorFlow Operations
    Overload 'connect' method
    """
    def __init__(self, shape=None, dtype=None, name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        
    def getInputShape(self, *args):
        return self.shape
        
    def getInputDtype(self, *args):
        return self.dtype
        
    def connect(self, prevLayerResult):
        self.validateInput(prevLayerResult)
        return self._connect(prevLayerResult)

def OpLayer(func, shape=None, dtype=None, name=None):
    return type(
        "CustomOpeartionLayer", 
        (_OpLayer,object), 
        {
            "_connect": lambda self, input: func(input) 
        })(shape=shape, dtype=dtype, name=name)       
        
class FeedForwardLayer(Layer):
    
    def __init__(self, features_out, features_in=None, dtype=None, name=None):
        self.features_out = features_out
        self.features_in = features_in
        self.dtype = dtype
        self.name = name
        
        self.varsCreated = False
        self.features_in_is_set = features_in != None;
        
    def setInput(self, features_in):
        if( self.features_in_is_set and self.features_in != features_in ):
            raise tfhu.TFHError(
                "{} setInput".format(self.name) ,
                "features_in is set twice and do not Match",
                "Previous Value : {}".format(self.features_in),
                "Current Value : {}".format(features_in))
        self.features_in = features_in
        
    def createVars(self):
        if( not self.varsCreated ):
            self.varsCreated = True
            
            # print(self.features_in, self.features_out)
            
            self.weight = init.weight_variable([self.features_in, self.features_out])
            self.bias = init.bias_variable([self.features_out])
        
    def connect(self, prevLayerResult):
        self.setInput(prevLayerResult._shape_as_list()[1])
        Layer.validateTFInput(self.name, prevLayerResult, shape=[None, self.features_in], dtype=self.dtype)
        self.createVars()
        return tf.matmul(prevLayerResult, self.weight) + self.bias
        
class ReshapeLayer(Layer):
    def __init__(self, shape, name=None):
        self.shape = shape
        self.name = name
        
    def connect(self, prevLayerResult):
        return tf.reshape(prevLayerResult, self.shape)