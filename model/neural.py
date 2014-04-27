import logging

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

from model.core import TranslationModel


class NeuralTranslationModel(TranslationModel):
    BIAS = True
    """True if the neural network should include bias units in its input
    and hidden layers."""

    HIDDEN_LAYER_SIZE = 100

    LEARNING_RATE = 0.075

    def __init__(self, source_vsm, target_vsm, bias=BIAS,
                 hidden_layer_size=HIDDEN_LAYER_SIZE,
                 learning_rate=LEARNING_RATE):
        """TODO document"""

        super(NeuralTranslationModel, self).__init__(source_vsm, target_vsm)

        self.network = None
        self.bias = bias
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate

    def train(self, seeds):
        input_size = self.vsm_source.layer1_size
        output_size = self.vsm_target.layer1_size
        network = buildNetwork(input_size, self.hidden_layer_size, output_size,
                               bias=self.bias, fast=True)

        dataset = SupervisedDataSet(input_size, output_size)
        for source_word, target_word in seeds:
            try:
                source = self.vsm_source[source_word]
            except KeyError:
                logging.warn(u'Source VSM missing word {}'.format(source_word))
                continue

            try:
                target = self.vsm_target[target_word]
            except KeyError:
                logging.warn(u'Target VSM missing word {}'.format(target_word))
                continue

            dataset.addSample(source, target)

        trainer = BackpropTrainer(network, dataset,
                                  learningrate=self.learning_rate)
        trainer.trainUntilConvergence()

    def translate_vec(self, source_vec, n=5):
        return self.network.activate(source_vec)
