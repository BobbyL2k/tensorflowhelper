import tensorflow as tf
# from enum import Enum

from . import utilities as tfhu


class Initializer():
    """Collection of Initializer Functions"""

    @staticmethod
    def weight_variable(shape):
        """Create weight variables of given shape
        The weight variables are randomized with SD of 0.1, making it
        sutable for Neural Networks
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """Create bias variables of given shape
        The baia variables are initialized with 0.1
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # def zeros_variable(shape):
    #     return tf.Variable(tf.zeros(shape))

    # def conv2d(x, W):
    #     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # def max_pool_2x2(x):
    #     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    #         strides=[1, 2, 2, 1], padding='SAME')

    # def max_pool_4x4(x):
    #     return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
    #         strides=[1, 4, 4, 1], padding='SAME')

# class initType(Enum):
#     zero = 1

class Layer(object):
    """Layer
    Abstract Class
    Layer with Inputs and Outputs
    """
    def __init__(self, name):
        self.name = name

    def connect(self, tf_input):
        """Connects the input to tha Layer and returns the output"""
        raise NotImplementedError

class ValidatableLayer(Layer):
    """Validatable Layer
    Abstract Class
    Layer with Inputs and Outputs that can be validated with input
    by using ValidatableLayer.validate_input
    """
    def __init__(self, name):
        Layer.__init__(self, name)

    def get_input_shape(self, *args):
        """Overloaded for Layer.validate_input()"""
        raise NotImplementedError

    def get_input_dtype(self, *args):
        """Overloaded for Layer.validate_input()"""
        raise NotImplementedError

    def validate_input(self, tf_layer_input):
        """Check if the Layer's shape and type match with ths input's"""
        tfhu.validate_tf_input(
            self.name,
            tf_layer_input,
            shape=self.get_input_shape(),
            dtype=self.get_input_dtype())

class ValidationLayer(ValidatableLayer):
    """ValidationLayer is used to check the current
    shape and dtype is matched with the your expectation

    shape -- shape of the expected tensor (default None --> matches everything)
    dtype -- data type of the expected tensor (default None --> matches everything)
    name  -- is for error message
    """
    def __init__(self, shape=None, dtype=None, name=None):
        ValidatableLayer.__init__(self, name)
        self.shape = shape
        self.dtype = dtype

    def get_input_shape(self, *args):
        return self.shape

    def get_input_dtype(self, *args):
        return self.dtype

    def connect(self, prevLayerResult):
        self.validate_input(prevLayerResult)
        return prevLayerResult

class OpLayer(ValidatableLayer):
    """Opeation Layer
    Layer for doing Native TensorFlow Operations
    func  -- is used to overload the '_connect' method
             no validation code is required as it is done automatically
    shape -- shape of the input tensor (default None --> matches everything)
             is used to do vaidation since TFH cannot automatically make sense
             of 'func'
    dtype -- data type of the input tensor (default None --> matches everything)
             is used to do vaidation since TFH cannot automatically make sense
             of 'func'
    name  -- is for error message
    """
    def __init__(self, func, shape=None, dtype=None, name=None):
        ValidatableLayer.__init__(self, name)
        self._connect = func
        self.shape = shape
        self.dtype = dtype

    def get_input_shape(self, *args):
        return self.shape

    def get_input_dtype(self, *args):
        return self.dtype

    def connect(self, prevLayerResult):
        self.validate_input(prevLayerResult)
        return self._connect(prevLayerResult)

class FeedForwardLayer(Layer):
    """FeedForward Neural Network Layer
    https://en.wikipedia.org/wiki/Feedforward_neural_network

    features_in  -- number of neuron (or features) in the previous layer
                    (default None --> will automatically match the previous layer)
    features_out -- number of neuron in this layer (number of features outputting)
    dtype        -- data type of the input/output tensor
                    (default None --> input : matches everything, output : matches input)
    name         -- is for error message
    """
    def __init__(self, features_out, features_in=None, dtype=None, name=None):
        Layer.__init__(self, name)
        self.features_out = features_out
        self.features_in = features_in
        self.dtype = dtype

        self.vars_created = False
        self.weight = None
        self.bias = None
        self.features_in_is_set = features_in != None

    def set_input(self, features_in):
        """Set the number of features_in
        Args:
            features_in: number of input features
        Raises:
            TFHError: if this method is called twice
        """
        if self.features_in_is_set and self.features_in != features_in:
            raise tfhu.TFHError(
                "{} set_input".format(self.name),
                "features_in is set twice and do not Match",
                "Previous Value : {}".format(self.features_in),
                "Current Value : {}".format(features_in))
        self.features_in = features_in

    def create_vars(self):
        """Create TensorFlow Placeholder variable"""
        if not self.vars_created:
            self.vars_created = True

            self.weight = Initializer.weight_variable([self.features_in, self.features_out])
            self.bias = Initializer.bias_variable([self.features_out])

    def connect(self, tf_input):
        self.set_input(tf_input._shape_as_list()[1])
        tfhu.validate_tf_input(
            self.name,
            tf_input,
            shape=[None, self.features_in],
            dtype=self.dtype)
        self.create_vars()
        return tf.matmul(tf_input, self.weight) + self.bias

class ReshapeLayer(Layer):
    """Reshape Layer
    uses tf.reshape(tensor, shape, name=None)
        tensor -- loaded with the layer input
        shape  -- from shape parameter

    shape -- shape of the output to resize to
    """
    def __init__(self, shape, name=None):
        Layer.__init__(self, name)
        self.shape = shape

    def connect(self, prevLayerResult):
        return tf.reshape(prevLayerResult, self.shape)
