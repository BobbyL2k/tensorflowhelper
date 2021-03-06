import tensorflow as tf
# from enum import Enum

from . import utilities as tfhu


class Initializer():
    """Collection of Initializer Functions"""

    @staticmethod
    def weight_variable(shape, name=None):
        """Create weight variables of given shape
        The weight variables are randomized with SD of 0.1, making it
        sutable for Neural Networks
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name="{}-weight".format(name))

    @staticmethod
    def bias_variable(shape, name=None):
        """Create bias variables of given shape
        The baia variables are initialized with 0.1
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name="{}-bias".format(name))

    @staticmethod
    def conv_weight_variable(width, height, depth_in, depth_out, name=None):
        return Initializer.weight_variable([width, height, depth_in, depth_out], name)

    @staticmethod
    def conv_bias_variable(depth, name=None):
        return Initializer.bias_variable([depth], name)

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

    Args:
        name -- is for error message
    """
    def __init__(self, name):
        self.name = name
        self._initialized = False

    def connect(self, tf_input):
        """Connects the input to tha Layer and returns the output"""
        raise NotImplementedError

    def get_tensorflow_variables(self):
        """Returns the all the TensorFlow variables that needs to be saved and loaded"""
        raise NotImplementedError

    def is_initialized_in(self, session):
        """Check if the TensorFlow variables are initialized"""
        if self._initialized:
            return True
        tf_var_list = self.get_tensorflow_variables()
        if len(tf_var_list) == 0:
            return True
        try:
            session.run(tf.assert_variables_initialized(self.get_tensorflow_variables()))
            self._initialized = True
            return True
        except tf.errors.FailedPreconditionError:
            return False

    def initialize_in(self, session):
        """Initialize TensorFlow variables in the provided session"""
        var_list = self.get_tensorflow_variables()

        tfhu.safe_initialize_in(var_list, session)

class ValidatableLayer(Layer):
    """Validatable Layer
    Abstract Class
    Layer with Inputs and Outputs that can be validated with input
    by using ValidatableLayer.validate_input

    Args:
        name -- is for error message
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

    Args:
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

    def get_tensorflow_variables(self):
        return []

class OpLayer(ValidatableLayer):
    """Opeation Layer
    Layer for doing Native TensorFlow Operations

    Args:
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
    def __init__(self, func, tf_variables=None, shape=None, dtype=None, name=None):

        if tf_variables is None:
            tf_variables = []

        ValidatableLayer.__init__(self, name)
        self._connect = func
        self.tf_variables = tf_variables
        self.shape = shape
        self.dtype = dtype

    def get_input_shape(self, *args):
        return self.shape

    def get_input_dtype(self, *args):
        return self.dtype

    def connect(self, prevLayerResult):
        self.validate_input(prevLayerResult)
        return self._connect(prevLayerResult)

    def get_tensorflow_variables(self):
        return self.tf_variables

class FeedForwardLayer(Layer):
    """FeedForward Neural Network Layer
    https://en.wikipedia.org/wiki/Feedforward_neural_network

    Args:
        features_out -- number of neuron in this layer (number of features outputting)
        features_in  -- number of neuron (or features) in the previous layer
                        (default None --> will automatically match the previous layer)
        dtype        -- data type of the input/output tensor
                        (default None --> input : matches everything, output : matches input)
        name         -- is for error message
    """
    def __init__(self, features_out, features_in=None, dtype=None, name=None):
        Layer.__init__(self, name)
        self.features_out = features_out
        self.features_in = features_in
        self.dtype = dtype

        self.features_in_is_set = features_in is not None

        self.tf_vars_created = False
        self.tf_weight = None
        self.tf_bias = None

    def set_input(self, features_in):
        """Set the number of features_in
        Args:
            features_in -- number of input features
        Raises:
            TFHError    -- if this method is called twice
        """
        if self.features_in_is_set and self.features_in != features_in:
            raise tfhu.TFHError(
                "{} set_input".format(self.name),
                "features_in is set twice and do not Match",
                "Previous Value : {}".format(self.features_in),
                "Current Value : {}".format(features_in))
        self.features_in = features_in

    def _create_tf_vars(self):
        """Create TensorFlow Placeholder variable"""
        if not self.tf_vars_created:
            self.tf_vars_created = True

            self.tf_weight = Initializer.weight_variable([self.features_in, self.features_out], name=self.name)
            self.tf_bias = Initializer.bias_variable([self.features_out], name=self.name)

    def connect(self, tf_input):
        self.set_input(tf_input._shape_as_list()[1])
        tfhu.validate_tf_input(
            self.name,
            tf_input,
            shape=[None, self.features_in],
            dtype=self.dtype)
        self._create_tf_vars()
        return tf.matmul(tf_input, self.tf_weight) + self.tf_bias

    def get_tensorflow_variables(self):
        return [
            self.tf_weight,
            self.tf_bias
        ]

class ReshapeLayer(Layer):
    """Reshape Layer
    uses tf.reshape(tensor, shape, name=None)
        tensor -- loaded with the layer input
        shape  -- from shape parameter

    Args:
        shape  -- shape of the output to resize to
    """
    def __init__(self, shape, name=None):
        Layer.__init__(self, name)
        self.shape = ReshapeLayer._cvt_none_to_mone(shape)

    @staticmethod
    def _cvt_none_to_mone(shape):
        if isinstance(shape, list):
            return list(map(ReshapeLayer._cvt_none_to_mone, shape))
        if shape is None:
            return -1
        else:
            return shape

    def connect(self, prevLayerResult):
        return tf.reshape(prevLayerResult, self.shape)

    def get_tensorflow_variables(self):
        return []

class ConvLayer(Layer):
    """Convolutional Neural Network Layer
    https://en.wikipedia.org/wiki/Convolutional_neural_network

    Args:
        depth_out     -- number of neuron in this layer (number of features outputting)
        kernel_width  -- width of the kernel
        kernel_height -- height of the kernel (default same value as width)
        depth_in      -- number of neuron (or features) in the previous conv-layer
                         (default None --> will automatically match the previous layer)
        padding       -- Apply padding inorder to make the width and height of the output
                         the same as the input (default True)
        dtype         -- data type of the input/output tensor
                         (default None --> input : matches everything, output : matches input)
        name          -- is for error message
    Note:
        The following variables if modified and the Layer can NOT be loaded with same model
            * depth_out
            * kernel_width
            * kernel_height
            * dtype
            * depth_in
        The following variables can be modified and the Layer can still be loaded
            * padding
            * the input tensor batch_count, width, and height
    """
    def __init__(self, depth_out, kernel_width, kernel_height=None, depth_in=None, padding=True, dtype=None, name=None):
        Layer.__init__(self, name)

        if kernel_height is None:
            kernel_height = kernel_width

        self.depth_out = depth_out
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.depth_in = depth_in
        self.padding = padding
        self.dtype = dtype

        self.depth_in_is_set = depth_in is not None

        self.tf_vars_created = False
        self.tf_weight = None

    def set_input(self, depth_in):
        """Set the number of depth_in
        Args:
            depth_in -- number of input features
        Raises:
            TFHError    -- if this method is called twice
        """
        if self.depth_in_is_set and self.depth_in != depth_in:
            raise tfhu.TFHError(
                "{} set_input".format(self.name),
                "depth_in is set twice and do not Match",
                "Previous Value : {}".format(self.depth_in),
                "Current Value : {}".format(depth_in))
        self.depth_in = depth_in

    def _create_tf_vars(self):
        """Create TensorFlow Placeholder variable"""
        if not self.tf_vars_created:
            self.tf_vars_created = True

            self.tf_weight = Initializer.conv_weight_variable(self.kernel_width, self.kernel_height,
                                                              self.depth_in, self.depth_out)

    def connect(self, tf_input):
        self.set_input(tf_input._shape_as_list()[3])
        tfhu.validate_tf_input(
            self.name,
            tf_input,
            shape=[None, None, None, self.depth_in],
            dtype=self.dtype)
        self._create_tf_vars()
        padding_cmd_str = "SAME" if self.padding else "VALID"
        return tf.nn.conv2d(tf_input, self.tf_weight, strides=[1, 1, 1, 1], padding=padding_cmd_str)

    def get_tensorflow_variables(self):
        return [self.tf_weight]

class ConvBiasLayer(Layer):
    """Bias for Convolutional Neural Network Layer
    https://en.wikipedia.org/wiki/Convolutional_neural_network

    Args:
        depth         -- number of neuron (or features) in the previous conv-layer
                         (default None --> will automatically match the previous layer)
        dtype         -- data type of the input/output tensor
                         (default None --> input : matches everything, output : matches input)
        name          -- is for error message
    Note:
        The following variables if modified and the Layer can NOT be loaded with same model
            * depth
            * dtype
        The following variables can be modified and the Layer can still be loaded
            * the input tensor batch_count, width, and height
    """
    def __init__(self, depth=None, dtype=None, name=None):
        Layer.__init__(self, name)

        self.depth = depth
        self.dtype = dtype

        self.depth_is_set = depth is not None

        self.tf_vars_created = False
        self.tf_bias = None

    def set_input(self, depth):
        """Set the number of depth
        Args:
            depth -- number of input features
        Raises:
            TFHError    -- if this method is called twice
        """
        if self.depth_is_set and self.depth != depth:
            raise tfhu.TFHError(
                "{} set_input".format(self.name),
                "depth is set twice and do not Match",
                "Previous Value : {}".format(self.depth),
                "Current Value : {}".format(depth))
        self.depth = depth

    def _create_tf_vars(self):
        """Create TensorFlow Placeholder variable"""
        if not self.tf_vars_created:
            self.tf_vars_created = True

            self.tf_bias = Initializer.conv_bias_variable(self.depth)

    def connect(self, tf_input):
        self.set_input(tf_input._shape_as_list()[3])
        tfhu.validate_tf_input(
            self.name,
            tf_input,
            shape=[None, None, None, self.depth],
            dtype=self.dtype)
        self._create_tf_vars()
        return tf.nn.bias_add(tf_input, self.tf_bias)

    def get_tensorflow_variables(self):
        return [self.tf_bias]

