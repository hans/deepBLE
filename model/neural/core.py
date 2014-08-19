import logging
import pickle
from pprint import pprint

import numpy as np

from model.core import TranslationModel


class NeuralTranslationModel(TranslationModel):
    """A translation model which trains a deep neural network on the
    given training examples.
    """

    BIAS = False
    """True if the neural network should include bias units in its input
    and hidden layers."""

    HIDDEN_LAYER_SIZE = 1000

    LEARNING_RATE = 0.001

    BATCH_SIZE = 10

    TRAIN_DEV_SPLIT = 0.9
    """Ratio of data input used for training (compared to total data
    input). Remaining data is held out for development testing."""

    def __init__(self, source_vsm, target_vsm, bias=BIAS,
                 hidden_layer_size=HIDDEN_LAYER_SIZE,
                 learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                 verbose=False):

        from pylearn2.costs import mlp as mlp_costs
        from pylearn2.costs.cost import SumOfCosts
        from pylearn2.models import mlp
        from pylearn2.train import Train
        from pylearn2.training_algorithms import sgd
        from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
        from pylearn2.termination_criteria import (EpochCounter, ChannelTarget,
                                                   MonitorBased)
        from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
        import theano

        super(NeuralTranslationModel, self).__init__(source_vsm, target_vsm)

        self.network = None

        self.bias = bias
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose

    def build_datasets(self, source_vecs, target_vecs):
        split = int(len(source_vecs) * self.TRAIN_DEV_SPLIT)

        X_train = np.mat(source_vecs[:split])
        Y_train = np.mat(target_vecs[:split])
        ds_train = DenseDesignMatrix(X=X_train, y=Y_train)

        X_dev = np.mat(source_vecs[split:])
        Y_dev = np.mat(target_vecs[split:])
        ds_dev = DenseDesignMatrix(X=X_dev, y=Y_dev)

        return ds_train, ds_dev

    def train_vecs(self, source_vecs, target_vecs):
        ds_train, ds_dev = self.build_datasets(source_vecs, target_vecs)

        # Determine visible layer dimensions
        input_size = self.source_vsm.layer1_size
        output_size = self.target_vsm.layer1_size

        # Hidden layer with sigmoid activation function
        hidden_layer = mlp.Sigmoid(layer_name='hidden', irange=.05, init_bias=1.,
                                   use_bias=self.bias,
                                   dim=self.hidden_layer_size)

        # Output layer with linear activation function
        output_layer = mlp.Linear(output_size, 'output', irange=.05,
                                  use_bias=self.bias)

        layers = [hidden_layer, output_layer]

        # Now construct neural network
        self.network = mlp.MLP(layers, nvis=input_size)

        # Build cost function
        error_cost = mlp_costs.Default()
        regularization_cost = mlp_costs.WeightDecay([1., 1.])
        cost_expr = SumOfCosts([error_cost])#, (.01, regularization_cost)])

        # Initialize SGD training algo
        algorithm = sgd.SGD(cost=cost_expr,
                            learning_rate=self.learning_rate,
                            batch_size=self.batch_size,
                            termination_criterion=EpochCounter(50000),#MonitorBased(.000001, N=5),
                            monitoring_dataset=ds_train)#cv)

        # Now build trainer

        # Extension: update learning rate based on cost changes
        extensions = [MonitorBasedLRAdjuster()]

        trainer = Train(model=self.network, dataset=ds_train,
                        algorithm=algorithm, extensions=extensions)

        # TODO constant-ize time budget
        trainer.main_loop()

        self.make_network_fn()

    def make_network_fn(self):
        X_sym = self.network.get_input_space().make_theano_batch()
        Y_sym = self.network.fprop(X_sym)
        self.network_fn = theano.function([X_sym], Y_sym)

    def load_object(self, obj):
        self.network = obj
        self.make_network_fn()

    def save_object(self):
        return self.network

    def translate_vec(self, source_vec):
        return self.network_fn(np.array([source_vec]))
