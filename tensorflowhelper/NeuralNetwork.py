import tensorflow as tf
import collections
from . import utilities as tfhu
from . import Layer


class NeuralNetwork(Layer):
    def __init__(self, layers = [], name = None):
        self.layers = layers
        self.name = name
        
        self.addNameToLayers(name)
                
    def addNameToLayers(self, name):
        for layer in self.layers:
            if isinstance(layer, NeuralNetwork):
                layer.addNameToLayers(str(name))
            else:
                layer.name = "{} {}".format(str(name), str(layer.name))
        
        
    def getInputShape(self, *args):
        if( len(self.layers) > 0 ):
            return self.layers[0].getInputShape()
        return None
        
    def getInputDtype(self, *args):
        if( len(self.layers) > 0 ):
            return self.layers[0].getInputDtype()
        return None
    
    def getOutputShape(self, sampleInput=None, *args):
        if( len(self.layers) > 0 ):
            return self.layers[-1].getOutputShape(sampleInput)
        return None
        
    def getOutputDtype(self, sampleInput=None, *args):
        if( len(self.layers) > 0 ):
            return self.layers[-1].getOutputDtype(sampleInput)
        return None
            
    def connect(self, prevLayerResult):
        for layer in self.layers:
            prevLayerResult = layer.connect(prevLayerResult)
        return prevLayerResult

class CostFunction:
    def meanSqErr(hypo, y):
        return tf.reduce_mean( tf.square( hypo - y ) )
    def crossEntropy(hypo, y):
        return tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) )

class Life(object):
    def __init__(self, neuralNetwork, costFunction=CostFunction.meanSqErr, optimizer=tf.train.AdamOptimizer(0.0001)):
        self.neuralNetwork = neuralNetwork
        self.session = tf.Session()
        self.costFunction = costFunction
        self.optimizer = optimizer
        self.tfvInputLayerPlaceholder  = None
        self.tfvOutputLayerPlaceholder = None
        self.tfvResult_pipe            = None
        self.tfvCost                   = None
        self.tfvTrainer                = None
        
    def connectNeuralNetwork(self, sampleInput, sampleOutput=None, willTrain=False ):
        self.tfvInputLayerPlaceholder = tf.placeholder( 
            sampleInput.dtype, 
            sampleInput.shape )
            
        self.neuralNetwork.validateInput(self.tfvInputLayerPlaceholder)
        
        self.tfvResult_pipe = self.neuralNetwork.connect( self.tfvInputLayerPlaceholder )
        
        if(willTrain):
            self.tfvOutputLayerPlaceholder = tf.placeholder( 
                sampleOutput.dtype, 
                sampleOutput.shape )
                
            self.neuralNetwork.validateOutput( self.tfvOutputLayerPlaceholder, sampleTfInput=self.tfvInputLayerPlaceholder )
            
            self.tfvCost = self.costFunction( self.tfvResult_pipe, self.tfvOutputLayerPlaceholder )
            
            self.tfvTrainer = self.optimizer.minimize( self.tfvCost )
        
    def initVar(self):
        self.session.run(tf.initialize_all_variables())
        
    def feed(self, inputLayerValue):
        return self.session.run( self.tfvResult_pipe, feed_dict={self.tfvInputLayerPlaceholder: inputLayerValue} )
        
    def train(self, inputLayerValue, outputLayerValue, processList=[]):
    
        processList.append(self.tfvCost)
        processList.append(self.tfvTrainer)
    
        result = self.session.run(processList, feed_dict={
            self.tfvInputLayerPlaceholder : inputLayerValue, 
            self.tfvOutputLayerPlaceholder: outputLayerValue})
            
        result.pop() # Remove None from Trainer
            
        return result
