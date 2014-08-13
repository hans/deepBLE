import logging
import pickle

import numpy as np
from numpy.linalg import pinv

from model import TranslationModel


def add_bias(matrix):
    """Add an affine transform "bias" row to a matrix."""

    return np.vstack([matrix, np.ones_like(matrix[0])])


class AffineTranslationModel(TranslationModel):
    """A translation model which learns an affine transform between the
    source and target spaces."""

    def __init__(self, source_vsm, target_vsm):
        super(AffineTranslationModel, self).__init__(source_vsm, target_vsm)

        # Matrix which performs the affine transform. Note that the
        # column space of this matrix is (d + 1)-dimensional -- shave
        # off the extra row before returning its outputs
        self.matrix = None

    def train_vecs(self, source_vecs, target_vecs):
        # Training matrices, where each column represents a single word
        # in the source or target language and each row represents a
        # dimension in the source or target VSM
        #
        # (Because this is an affine transform, we have an extra bias
        # row at the bottom of the matrix)
        source_matrix = add_bias(np.matrix(source_vecs).transpose())
        target_matrix = add_bias(np.matrix(target_vecs).transpose())

        # We have $F = AE$, where $F$ is our target matrix (each column
        # is a word embedding), $E$ is our source matrix, and $A$ is our
        # translation matrix.
        #
        # Then $A = F E^{-1}$.
        self.matrix = target_matrix * pinv(source_matrix)

    def translate_vec(self, source_vec):
        if self.matrix is None:
            raise RuntimeError("Model not yet trained")

        # Add bias row
        source_vec = np.append(source_vec, 1)

        out = np.squeeze(np.asarray(self.matrix.dot(source_vec)))

        # Remove bias row
        return out[:-1]

    def load_object(self, obj):
        self.matrix = obj

    def save_object(self):
        return self.matrix
