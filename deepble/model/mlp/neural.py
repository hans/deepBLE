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


class NegativeRectifiedLinear(mlp.RectifiedLinear):
    """
    An abstract rectified linear MLP layer which transforms its inputs
    / outputs so as to accommodate negative values.

    See implementations `NegatingRectifiedLinear`,
    `NegativeSwitchRectifiedLinear`.
    """

    def __init__(self, **kwargs):
        super(NegativeRectifiedLinear, self).__init__(**kwargs)

    @wraps(mlp.Layer.set_input_space)
    def set_input_space(self, space):
        super(NegativeRectifiedLinear, self).set_input_space(space)

        # Build negating matrix now that we know our layer dimensions

        out_dim = self.get_output_space().get_total_dimension()

        num_positive = out_dim / 2
        num_negative = out_dim - num_positive

        self.modifier = theano.shared(np.diag(np.concatenate((
            np.ones(num_positive), -1 * np.ones(num_negative)))))

    @wraps(mlp.Layer.fprop)
    def fprop(self, *args, **kwargs):
        raise NotImplementedError("abstract method")


class NegatingRectifiedLinear(NegativeRectifiedLinear):
    """
    Rectified linear MLP layer which negates its activation function
    on half of its neurons (Glorot and Bengio 2011).

    Concretely, for half of our neurons the activation function is a
    normal ReLU:

        \[ f_1(x) = \text{max}(0, x) \]

    for the remaining half we negate the outputs:

        \[ f_2(x) = -\text{max}(0, x) \]
    """

    def __init__(self, **kwargs):
        super(NegatingRectifiedLinear, self).__init__(**kwargs)

    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):
        p = super(NegatingRectifiedLinear, self).fprop(state_below)

        # p is n_samples x n_nodes
        return T.dot(p, self.modifier)


class NegativeSwitchRectifiedLinear(NegativeRectifiedLinear):
    """
    Rectified linear MLP layer which accepts negative rather than
    positive values on half of its neurons.

    Concretely, the activation function here is

       \[ f_1(x) = \text{max}(0, x) \]

    for one half of the nodes (a normal ReLU) and

       \[ f_2(x) = \text{min}(0, x) \]

    for the other half of the nodes.

    Note the difference between this class and
    `NegatingRectifiedLinear`.
    """

    def __init__(self, **kwargs):
        super(NegativeSwitchRectifiedLinear, self).__init__(**kwargs)

    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):
        # Transform one half of the inputs so that our ReLU max(0, x) is
        # actually doing a min(0, x) (just negate the values)
        state_below = T.dot(state_below, self.modifier)

        # Now feed forward with the standard ReLU
        p = super(NegativeSwitchRectifiedLinear, self).fprop(state_below)

        # Un-negate the half of the inputs we negated earlier
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
