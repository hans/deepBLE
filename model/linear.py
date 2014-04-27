import numpy as np
from numpy.linalg import pinv

from model import TranslationModel


class LinearTranslationModel(TranslationModel):
    def __init__(self, source_vsm, target_vsm):
        super(LinearTranslationModel, self).__init__(source_vsm, target_vsm)
        self.matrix = None

    def train(self, seeds):
        source_matrix = np.matrix([self.source_vsm[source_word]
                                   for source_word, _ in seeds])
        target_matrix = np.matrix([self.target_vsm[target_word]
                                   for _, target_word in seeds])

        # We have $F = AE$, where $F$ is our target matrix (each column
        # is a word embedding), $E$ is our target matrix, and $A$ is our
        # translation matrix.
        #
        # Then $A = F E^{-1}$.
        self.matrix = target_matrix * pinv(source_matrix)

    def translate_vec(self, source_vec, n=5):
        if self.matrix is None:
            raise RuntimeError("Model not yet trained")

        return self.matrix * source_vec
