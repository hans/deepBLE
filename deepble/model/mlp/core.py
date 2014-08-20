"""
Defines the multilayer perceptron framework that our neural network
model and other translation models will use.

Note that most of the Pylearn and Theano imports are lazy-loaded
within the functions / classes that require them -- this is to avoid
annoying load times during non-MLP-related testing.
"""

import numpy as np
from pylearn2.costs import cost
from pylearn2.models import mlp
import theano.tensor as T

from deepble.model.core import TranslationModel


def get_dataset(which):
    """
    Hacky way to allow us to leave the dataset unspecified in a
    Pylearn config.

    This function is just assigned some state from a running
    `MLPTranslationModel`, and provides that state to Pylearn when
    requested. Note that this is not thread-safe among multiple
    MLP models which may be running in the same process!
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


class MLPTranslationModel(TranslationModel):
    """
    A translation model which trains an arbitary MLP on the given data.

    The MLP model is provided via a Pylearn configuration file as an
    argument to the class constructor.
    """

    TRAIN_DEV_SPLIT = 0.9
    """Ratio of data input used for training (compared to total data
    input). Remaining data is held out for development testing."""

    def __init__(self, source_vsm, target_vsm, config_file, verbose=False):
        super(MLPTranslationModel, self).__init__(source_vsm, target_vsm)

        self.network = None
        self.network_fn = None

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
