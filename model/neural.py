import logging
import pickle
from pprint import pprint

import numpy as np
from pylearn2.costs import mlp as mlp_costs
from pylearn2.costs.cost import SumOfCosts
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import theano

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

    BATCH_SIZE = 10

    def __init__(self, source_vsm, target_vsm, bias=BIAS,
                 hidden_layer_size=HIDDEN_LAYER_SIZE,
                 learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                 verbose=False):
        """TODO document"""

        super(NeuralTranslationModel, self).__init__(source_vsm, target_vsm)

        self.network = None

        self.bias = bias
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose

    def train_vecs(self, source_vecs, target_vecs):
        # Build dataset
        X = np.mat(source_vecs)
        Y = np.mat(target_vecs)
        dataset = DenseDesignMatrix(X=X, y=Y)

        # Determine visible layer dimensions
        input_size = self.source_vsm.layer1_size
        output_size = self.target_vsm.layer1_size

        # Hidden layer with sigmoid activation function
        hidden_layer = mlp.Sigmoid(layer_name='hidden', irange=.1, init_bias=1.,
                                   use_bias=self.bias,
                                   dim=self.hidden_layer_size)

        # Output layer with linear activation function
        output_layer = mlp.Linear(output_size, 'output', irange=.1,
                                  use_bias=self.bias)

        layers = [hidden_layer, output_layer]

        # Now construct neural network
        self.network = mlp.MLP(layers, nvis=input_size)

        # Build cost function
        error_cost = mlp_costs.Default()
        regularization_cost = mlp_costs.WeightDecay([1., 1.])
        cost_expr = SumOfCosts([error_cost])# , (.5, regularization_cost)])

        # Initialize SGD trainer
        trainer = sgd.SGD(cost=cost_expr,
                          learning_rate=self.learning_rate,
                          batch_size=self.batch_size,
                          termination_criterion=MonitorBased(.0000001, N=100),
                          monitoring_dataset=dataset)
        trainer.setup(self.network, dataset)

        while True:
            trainer.train(dataset=dataset)

            self.network.monitor.report_epoch()
            self.network.monitor()

            if not trainer.continue_learning(self.network):
                break

        X_sym = self.network.get_input_space().make_theano_batch()
        Y_sym = self.network.fprop(X_sym)
        self.network_fn = theano.function([X_sym], Y_sym)

    def load(self, path):
        with open(path, 'r') as model_f:
            self.network = pickle.load(model_f)

    def save(self, path):
        logging.info("Saving neural network model to '{}'".format(path))
        with open(path, 'w') as model_f:
            pickle.dump(self.network, model_f)

    def translate_vec(self, source_vec):
        return self.network_fn(np.array([source_vec]))
