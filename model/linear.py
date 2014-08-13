import logging
import pickle

import numpy as np
from numpy.linalg import pinv

from model import TranslationModel


class LinearTranslationModel(TranslationModel):
    def __init__(self, source_vsm, target_vsm):
        super(LinearTranslationModel, self).__init__(source_vsm, target_vsm)
        self.matrix = None

    def train_vecs(self, source_vecs, target_vecs):
        # Training matrices, where each column represents a single word
        # in the source or target language and each row represents a
        # dimension in the source or target VSM
        source_matrix = np.matrix(source_vecs).transpose()
        target_matrix = np.matrix(target_vecs).transpose()

        # We have $F = AE$, where $F$ is our target matrix (each column
        # is a word embedding), $E$ is our source matrix, and $A$ is our
        # translation matrix.
        #
        # Then $A = F E^{-1}$.
        self.matrix = target_matrix * pinv(source_matrix)

    def translate_vec(self, source_vec):
        if self.matrix is None:
            raise RuntimeError("Model not yet trained")

        return self.matrix.dot(source_vec)

    def load_object(self, obj):
        self.matrix = obj

    def save_object(self):
        return self.matrix
