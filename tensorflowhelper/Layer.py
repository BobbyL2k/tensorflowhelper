import tensorflow as tf
from enum import Enum
from . import utilities as tfhu

# class init():
#     def zeros_variable(shape):
#         return tf.Variable(tf.zeros(shape))
        
#     def weight_variable(shape):
#         initial = tf.truncated_normal(shape, stddev=0.1)
#         return tf.Variable(initial)

#     def bias_variable(shape):
#         initial = tf.constant(0.1, shape=shape)
#         return tf.Variable(initial)

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
    
    def getOutputShape(self, sampleTfInput=None, *args):
        """Overloaded for Layer.validateOutput()"""
        raise NotImplementedError
        
    def getOutputDtype(self, sampleTfInput=None, *args):
        """Overloaded for Layer.validateOutput()"""
        raise NotImplementedError
            
    def connect(self, *args):
        raise NotImplementedError
        
    def validateInput(self, layerInput):
        """Check if the Layer's shape and type match with ths input's"""
        tfhu.validate(self.name, "Tensor shape", self.getInputShape(), layerInput._shape_as_list())
        tfhu.validate(self.name, "Tensor dtype", self.getInputDtype(), layerInput.dtype)
        
    def validateOutput(self, layerOutput, sampleTfInput=None):
        """Check if the Layer's shape and type match with ths Output's"""
        tfhu.validate(self.name, "Tensor shape", self.getOutputShape(sampleTfInput), layerOutput._shape_as_list())
        tfhu.validate(self.name, "Tensor dtype", self.getOutputDtype(sampleTfInput), layerOutput.dtype)

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
    
    def getOutputShape(self, sampleTfInput):
        if(sampleTfInput == None):
            return self.shape
        else:
            return sampleTfInput._shape_as_list();
        
    def getOutputDtype(self, sampleTfInput):
        if(sampleTfInput == None):
            return self.dtype
        else:
            return sampleTfInput.dtype
        
    def connect(self, prevLayerResult):
        raise NotImplementedError

def OpLayer(func, name=None):
    return type(
        "CustomOpeartionLayer", 
        (_OpLayer,object), 
        {
            "connect": lambda self, input: func(input) 
        })()       
        
class FeedForwardLayer(Layer):
    def __init__(self, features_out, features_in=None):
        self.features_out = features_out
        self.features_in = features_in
        self.varsCreated = False
        
    def getOutputShape(self, *args):
        return self.shape
        
    def getOutputDtype(self, *args):
        return self.dtype
        
    # def setInput(self, features_in):
    #     self.features_in = features_in
        
    def createVars(self):
        if( not self.createVars ):
            self.varsCreated = True
            
            self.weight = init.weight_variable([self.features_in, self.features_out])
            self.bias = init.bias_variable([self.features_out])
        
    def connect(self, prevLayerResult):
        self.createVars()
        return tf.matmul(prevLayerResult, self.weight) + self.bias
