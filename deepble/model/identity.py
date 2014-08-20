from deepble.model import TranslationModel


class IdentityTranslationModel(TranslationModel):
    """Treats the translation of any input vector as exactly that input
    vector. (Assumes the dimensionality of the input and output spaces
    match.)"""

    def __init__(self, source_vsm, target_vsm):
        super(IdentityTranslationModel, self).__init__(source_vsm, target_vsm)

    def train_vecs(self, source_vecs, target_vecs):
        # No training necessary!
        pass

    def translate_vec(self, source_vec):
        return source_vec
