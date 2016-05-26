import tensorflow as tf
from enum import Enum
from . import utilities as tfhu

# class TensorFlow()

class init():
    def zeros_variable(shape):
        return tf.Variable(tf.zeros(shape))
        
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
            strides=[1, 2, 2, 1], padding='SAME')
            
    def max_pool_4x4(x):
        return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], 
            strides=[1, 4, 4, 1], padding='SAME')

class initType(Enum):
    zero = 1
    
class Layer(object):
    """
    Abstract Class
    Layer with Inputs and Outputs
    """
    
    def getInputShape(self, *args):
        raise NotImplementedError
        
    def getInputDtype(self, *args):
        raise NotImplementedError
    
    def getOutputShape(self, *args):
        raise NotImplementedError
        
    def getOutputDtype(self, *args):
        raise NotImplementedError
            
    def connect(self, *args):
        raise NotImplementedError
        
    def validatieInput(self, layerInput):
        """Check if the Layer's shape and type match with ths input's"""
        tfhu.validatie(self.name, "Tensor shape", self.getInputShape(), layerInput._shape_as_list())
        tfhu.validatie(self.name, "Tensor dtype", self.getInputDtype(), layerInput.dtype)
        
    def validatieOutput(self, layerOutput):
        """Check if the Layer's shape and type match with ths Output's"""
        tfhu.validatie(self.name, "Tensor shape", self.getOutputShape(), layerOutput._shape_as_list())
        tfhu.validatie(self.name, "Tensor dtype", self.getOutputDtype(), layerOutput.dtype)
            

# class ILayer(Layer):
#     """
#     Abstract Class
#     Layer with Inputs
#     """
#     def setInput(self, *args):
#         raise NotImplementedError
        
# class OLayer(Layer):
#     """
#     Abstract Class
#     Layer with Output
#     """
#     def getOutput(self, *args):
#         raise NotImplementedError

class ValidationLayer(Layer):
    def __init__(self, shape=None, dtype=None, name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        
    def getInputShape(self, *args):
        return self.shape
        
    def getInputDtype(self, *args):
        return self.dtype
    
    def getOutputShape(self, *args):
        return self.shape
        
    def getOutputDtype(self, *args):
        return self.dtype
        
    def connect(self, prevLayerResult):
        self.validatieInput(prevLayerResult)
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
    
    def getOutputShape(self, *args):
        return self.shape
        
    def getOutputDtype(self, *args):
        return self.dtype
        
    def connect(self, prevLayerResult):
        raise NotImplementedError

def OpLayer(func, name=None):
    return type(
        "CustomOpeartionLayer", 
        (_OpLayer,object), 
        {
            "connect": lambda self, input: func(input) 
        })()       
        
# class FeedForwardLayer(ILayer, OLayer):
#     def __init__(self, features_out, features_in=None):
#         self.features_out = features_out
#         self.features_in = features_in
        
#     def setInput(self, features_in):
#         self.features_in = features_in
    
#         self.weight = init.weight_variable([self.features_in, self.features_out])
#         self.bias = init.bias_variable([self.features_out])
        
#     def connect(self, prevLayerResult):
#         return prevLayerResult
