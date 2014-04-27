import logging

import numpy as np
from numpy.linalg import pinv

from model import TranslationModel


class LinearTranslationModel(TranslationModel):
    def __init__(self, source_vsm, target_vsm):
        super(LinearTranslationModel, self).__init__(source_vsm, target_vsm)
        self.matrix = None

    def train(self, seeds):
        source_vectors = []
        target_vectors = []
        for source_word, target_word in seeds:
            if source_word not in self.source_vsm:
                logging.warn(u"Training word '{}' not present in source VSM"
                             .format(source_word))
                continue
            elif target_word not in self.target_vsm:
                logging.warn(u"Training word '{}' not present in target VSM"
                             .format(target_word))
                continue

            source_vectors.append(self.source_vsm[source_word])
            target_vectors.append(self.target_vsm[target_word])

        # Training matrices, where each column represents a single word
        # in the source or target language and each row represents a
        # dimension in the source or target VSM
        source_matrix = np.matrix(source_vectors).transpose()
        target_matrix = np.matrix(target_vectors).transpose()

        # We have $F = AE$, where $F$ is our target matrix (each column
        # is a word embedding), $E$ is our target matrix, and $A$ is our
        # translation matrix.
        #
        # Then $A = F E^{-1}$.
        self.matrix = target_matrix * pinv(source_matrix)

    def translate_vec(self, source_vec, n=5):
        if self.matrix is None:
            raise RuntimeError("Model not yet trained")

        return self.matrix.dot(source_vec)
