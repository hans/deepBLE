"""
Defines the framework for the neural-network translation model
experiments.

Note that most of the Pylearn and Theano imports are lazy-loaded
within the functions / classes that require them -- this is to avoid
annoying load times during non-NN-related testing.
"""

import logging
import pickle
from pprint import pprint

import numpy as np
from pylearn2.costs import cost
from pylearn2.models import mlp
from pylearn2.utils import wraps
import theano
import theano.tensor as T

from deepble.model.core import TranslationModel


def get_dataset(which):
    """
    Hacky way to allow us to leave the dataset unspecified in the
    Pylearn config.

    This function is just assigned some state from a running
    NeuralTranslationModel, and provides that state to Pylearn when
    requested. Note that this is not thread-safe among multiple
    neural models which may be running in the same process!
    """

    return getattr(get_dataset, which)


class MeanSquaredErrorCost(cost.DefaultDataSpecsMixin, cost.Cost):
    """
    Mean squared error:

    MSE = 1/n \sum (Yhat - Y)^2
    """

    supervised = True

    def __init__(self, *args, **kwargs):
        super(MeanSquaredErrorCost, self).__init__(*args, **kwargs)

    def expr(self, model, data, **kwargs):
        """
        Calculate the MSE for the given data.
        """

        self.get_data_specs(model)[0].validate(data)

        X, Y = data
        Yhat = model.fprop(X)

        return T.sqr(Yhat - Y).mean()


class NegatingRectifiedLinear(mlp.RectifiedLinear):
    """
    Rectified linear MLP layer which negates its activation function
    on half of its neurons (Glorot and Bengio 2011).
    """

    def __init__(self, **kwargs):
        super(NegatingRectifiedLinear, self).__init__(**kwargs)

    @wraps(mlp.Layer.set_input_space)
    def set_input_space(self, space):
        super(NegatingRectifiedLinear, self).set_input_space(space)

        # Build negating matrix now that we know our layer dimensions

        out_dim = self.get_output_space().get_total_dimension()

        num_positive = out_dim / 2
        num_negative = out_dim - num_positive

        self.modifier = theano.shared(np.diag(np.concatenate((
            np.ones(num_positive), -1 * np.ones(num_negative)))))

    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):
        p = super(NegatingRectifiedLinear, self).fprop(state_below)

        # p is n_samples x n_nodes
        return T.dot(p, self.modifier)


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

    def __init__(self, source_vsm, target_vsm,
                 config_file='deepble/model/mlp/config/mlp.yaml',
                 verbose=False):
        super(NeuralTranslationModel, self).__init__(source_vsm, target_vsm)

        self.network = None
        self.network_fn = None

        # TODO make configurable
        self.network_cfg = config_file

        self.verbose = verbose

    def build_datasets(self, source_vecs, target_vecs):
        from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

        split = int(len(source_vecs) * self.TRAIN_DEV_SPLIT)

        X_train = np.mat(source_vecs[:split])
        Y_train = np.mat(target_vecs[:split])
        ds_train = DenseDesignMatrix(X=X_train, y=Y_train)

        X_dev = np.mat(source_vecs[split:])
        Y_dev = np.mat(target_vecs[split:])
        ds_dev = DenseDesignMatrix(X=X_dev, y=Y_dev)

        # Assign to global object
        get_dataset.train = ds_train
        get_dataset.dev = ds_dev

    def train_vecs(self, source_vecs, target_vecs):
        self.build_datasets(source_vecs, target_vecs)

        from pylearn2.space import VectorSpace
        from pylearn2.utils.serial import load_train_file

        # TODO allow overrides via parameters
        train = load_train_file(self.network_cfg)

        # Change input layer size
        model = train.model
        if not isinstance(model, mlp.MLP):
            raise RuntimeError("Provided network config does not use "
                               "MLP model -- not supported by this "
                               "translation model code")

        input_size = self.source_vsm.layer1_size
        model.set_input_space(VectorSpace(dim=input_size))

        # Change output layer size
        final_layer = model.layers[-1]
        if not isinstance(final_layer, mlp.Linear):
            raise RuntimeError("Provided network config does not have "
                               "a linear output layer -- not supported "
                               "by this translation model code")

        output_size = self.target_vsm.layer1_size
        # TODO is this sufficient for the linear layer? Might need to
        # call some setter which updates internal state I don't
        # understand
        final_layer.dim = output_size

        # Now begin training
        train.main_loop()

        self.network = train.model

        self.make_network_fn()

    def make_network_fn(self):
        import theano

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
