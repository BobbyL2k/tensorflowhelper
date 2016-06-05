import tensorflow as tf
from . import utilities as tfhu

class CostFunction:
    """Collection of Cost Functions to be minimized"""
    @staticmethod
    def mean_sq_err(hypo, actual_value):
        """Calculate Mean Square Error
        Args:
            hypo         -- TensorFlow variable of the hypothesis
            actual_value -- TensorFlow variable of the expected value
        Returns:
            TensorFlow variable of the Mean Square Error
        """
        return tf.reduce_mean(tf.square(hypo - actual_value))
    @staticmethod
    def cross_entropy(hypo, actual_value):
        """Calculate Cross Entropy
        Args:
            hypo         -- TensorFlow variable of the hypothesis
            actual_value -- TensorFlow variable of the expected value
        Returns:
            TensorFlow variable of the Cross Entropy
        """
        return tf.reduce_mean(
            -tf.reduce_sum(actual_value * tf.log(hypo), reduction_indices=[1]))

class Life(object):
    """NeuralNetwork Manager
    Life helps with session management, connecting, feeding, and training NeuralNetworks
    """
    def __init__(self,
                 neural_network,
                 cost_function=CostFunction.mean_sq_err,
                 optimizer=tf.train.AdamOptimizer(0.0001)):

        self.session = tf.Session()

        self.neural_network = neural_network
        self.cost_function = cost_function
        self.optimizer = optimizer

        self.tfvInputLayerPlaceholder  = None
        self.tfvOutputLayerPlaceholder = None
        self.tfvResult_pipe            = None
        self.tfvCost                   = None
        self.tfvTrainer                = None

    def connect_neural_network(self, sample_input, sample_output=None, will_train=False):
        """Connects the NeuralNetworks inside to a TensorFLow placeholder variable
        Args:
            sample_input  -- A sample of the input, the dtype and shape is
                             used to form the TensorFlow placeholder variable
            sample_output -- A sample of the ouput for running the hypothesis against,
                             the dtype and shape is used to form the TensorFlow
                             placeholder variable, this is required if will_train=True
                             (default None)
            will_train    -- True if Life will be used to train in the future
        Returns:
            None
        """
        self.tfvInputLayerPlaceholder = tf.placeholder(
            sample_input.dtype,
            sample_input.shape)

        self.neural_network.validate_input(self.tfvInputLayerPlaceholder)

        self.tfvResult_pipe = self.neural_network.connect(self.tfvInputLayerPlaceholder)

        if will_train:
            tfhu.validate_tf_input(
                "Life Output",
                self.tfvResult_pipe,
                shape=sample_output.shape,
                dtype=sample_output.dtype)

            self.tfvOutputLayerPlaceholder = tf.placeholder(
                sample_output.dtype,
                sample_output.shape)

            self.tfvCost = self.cost_function(self.tfvResult_pipe, self.tfvOutputLayerPlaceholder)

            tf_vars_wo_optimizer = set(tf.all_variables())
            self.tfvTrainer = self.optimizer.minimize(self.tfvCost)
            self.session.run(
                tf.initialize_variables(set(tf.all_variables()) - tf_vars_wo_optimizer))


    def init_var(self, var_list=None):
        """Initialize All Variables
        Initialize all variables for the managed session if list
        not specified else the variables in the list is initialized
        Note: Initializing all variables will overwirte any loaded
              variables
        Args:
            var_list -- list of variables to initialize
            (default None --> all variables)
        """
        if var_list is None:
            var_list = self.neural_network.get_tensorflow_variables()

        tfhu.safe_initialize_in(var_list, self.session)

    def init_network(self, network_list=None):
        """Initialize Variables inside a network/layer
        calls tf.initialize_all_variables() for the self.session"""
        for network in network_list:
            network.initialize_in(self.session)

    @staticmethod
    def _create_saver_dict(tf_var_list):
        return dict(zip(
            [str(index) for index in range(len(tf_var_list))],
            tf_var_list))

        # return tf_var_list

    def load_saved_model(self, path, network=None):
        """Loads the specified network from a file in the specified path
        Args:
            path    -- is the path to load the Network model from (includes file extension)
            network -- is the network (must be inside Life object) specified to be loaded
                       with initialization values
                       (default: Root Network given at Life Initialization)
        """

        if network is None:
            network = self.neural_network

        tf_var_list = network.get_tensorflow_variables()
        if len(tf_var_list) == 0:
            raise tfhu.TFHError(
                "Load model failed because network doesn't have any variable to load")

        tfhu.warn_if_initialized(tf_var_list, self.session)
        saver = tf.train.Saver(Life._create_saver_dict(tf_var_list))
        saver.restore(self.session, path)
        print("Network {} loaded from file: {}".format(network.name, path))

    def save_current_model(self, path, network=None):
        """Saves the specified network to a file in the specified path
        Args:
            path    -- is the path to save the Network model at (includes file extension)
            network -- is the network (must be inside Life object) specified to be save
                       (default: Root Network given at Life Initialization)
        """

        if network is None:
            network = self.neural_network

        if not network.is_initialized_in(self.session):
            raise tfhu.TFHError("Save model failed because network is not initialized")

        tf_var_list = network.get_tensorflow_variables()
        if len(tf_var_list) == 0:
            raise tfhu.TFHError(
                "Save model failed because network doesn't have any variable to save")
        saver = tf.train.Saver(Life._create_saver_dict(tf_var_list))
        save_path = saver.save(self.session, path)
        print("Network {} saved in file: {}".format(network.name, save_path))

    def feed(self, input_layer_value):
        """Feed NeuralNetwork with input
        Args:
            input_layer_value -- input for the NeuralNetwork
        Returns:
            Output from the NeuralNetwork
        """
        return self.session.run(
            self.tfvResult_pipe,
            feed_dict={
                self.tfvInputLayerPlaceholder: input_layer_value})

    def train(self, input_layer_value, output_layer_value, process_list=None):
        """Train NeuralNetwork assigned with input and expected output"""

        if process_list is None:
            process_list = []

        process_list.append(self.tfvCost)
        process_list.append(self.tfvTrainer)

        result = self.session.run(process_list, feed_dict={
            self.tfvInputLayerPlaceholder : input_layer_value,
            self.tfvOutputLayerPlaceholder: output_layer_value})

        result.pop() # Remove None from tfvTrainer

        return result
