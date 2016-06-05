import tensorflow as tf
from . import ValidatableLayer


class NeuralNetwork(ValidatableLayer):
    def __init__(self, layers=None, name=None):

        ValidatableLayer.__init__(self, name)

        if layers is None:
            layers = []

        self.layers = layers

        self._add_name_to_layers(name)

    def _add_name_to_layers(self, name):
        """Adds name to all of the NeuralNetwork/Layers inside the NeuralNetwork"""
        for layer in self.layers:
            if isinstance(layer, NeuralNetwork):
                layer._add_name_to_layers(str(name))
            else:
                layer.name = "{}-{}".format(str(name), str(layer.name))


    def get_input_shape(self, *args):
        if len(self.layers) > 0 and isinstance(self.layers[0], ValidatableLayer):
            return self.layers[0].get_input_shape()
        return None

    def get_input_dtype(self, *args):
        if len(self.layers) > 0 and isinstance(self.layers[0], ValidatableLayer):
            return self.layers[0].get_input_dtype()
        return None

    def connect(self, prev_layer_result):
        for layer in self.layers:
            prev_layer_result = layer.connect(prev_layer_result)
        return prev_layer_result

    def get_tensorflow_variables(self):
        result = []
        for layer in self.layers:
            result.extend(layer.get_tensorflow_variables())
        return result