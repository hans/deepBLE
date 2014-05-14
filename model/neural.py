import logging

import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.xml.networkwriter import NetworkWriter

from model.core import TranslationModel


class NeuralTranslationModel(TranslationModel):
    """A translation model which trains a deep neural network on the
    given training examples.
    """

    BIAS = True
    """True if the neural network should include bias units in its input
    and hidden layers."""

    HIDDEN_LAYER_SIZE = 1000

    LEARNING_RATE = 0.01

    def __init__(self, source_vsm, target_vsm, bias=BIAS,
                 hidden_layer_size=HIDDEN_LAYER_SIZE,
                 learning_rate=LEARNING_RATE, verbose=False):
        """TODO document"""

        super(NeuralTranslationModel, self).__init__(source_vsm, target_vsm)

        self.network = None

        self.bias = bias
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.verbose = verbose

    def train_vecs(self, source_vecs, target_vecs):
        input_size = self.source_vsm.layer1_size
        output_size = self.target_vsm.layer1_size
        self.network = buildNetwork(input_size, self.hidden_layer_size,
                                    output_size, bias=self.bias, fast=True)

        dataset = SupervisedDataSet(input_size, output_size)
        dataset.setField('input', np.mat(source_vecs))
        dataset.setField('target', np.mat(target_vecs))

        trainer = BackpropTrainer(self.network, dataset,
                                  learningrate=self.learning_rate,
                                  verbose=self.verbose)
        trainer.trainUntilConvergence()

    def load(self, path):
        self.network = NetworkReader.readFrom(path)

    def save(self, path):
        logging.info("Saving neural network model to '{}'".format(path))
        NetworkWriter.writeToFile(self.network, path)

    def translate_vec(self, source_vec):
        return self.network.activate(source_vec)
