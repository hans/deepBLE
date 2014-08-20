"""
Defines the framework for the neural-network translation model
experiments.

Note that most of the Pylearn and Theano imports are lazy-loaded
within the functions / classes that require them -- this is to avoid
annoying load times during non-NN-related testing.
"""

import numpy as np
from pylearn2.models import mlp
from pylearn2.utils import wraps
import theano
import theano.tensor as T

from deepble.model.core import TranslationModel
from deepble.model.mlp import MLPTranslationModel


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


class NeuralTranslationModel(MLPTranslationModel):
    """
    A translation model which trains a neural network on the given data.
    """

    def __init__(self, source_vsm, target_vsm,
                 config_file='deepble/model/mlp/config/neural.yaml',
                 **kwargs):
        super(NeuralTranslationModel, self).__init__(source_vsm, target_vsm,
                                                     config_file, **kwargs)
