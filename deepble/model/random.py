from numpy.random import random

from deepble.model import TranslationModel


class RandomTranslationModel(TranslationModel):
    # Lower and upper bounds for randomly generated values
    LOWER_BOUND = -1.8777409
    UPPER_BOUND = 1.8641127

    def __init__(self, source_vsm, target_vsm, lower_bound=LOWER_BOUND,
                 upper_bound=UPPER_BOUND):
        super(RandomTranslationModel, self).__init__(source_vsm, target_vsm)

        self.dim = self.target_vsm.layer1_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def train_vecs(self, source_vecs, target_vecs):
        # No training necessary!
        pass

    def translate_vec(self, source_vec):
        return ((self.upper_bound - self.lower_bound)
                * random(self.dim)
                + self.lower_bound)
