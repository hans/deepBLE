from scipy.spatial import distance


class TranslationModel(object):
    """A model which uses paired vector-space language models to translate
    words between two languages.
    """

    def __init__(self, source_vsm, target_vsm):
        self.source_vsm = source_vsm
        self.target_vsm = target_vsm

    def train(self, seeds):
        """Train the model on a seed set, a list of pairs of the form

            (source_word, translationally_equivalent_target_word)
        """

        raise NotImplementedError("abstract method")

    def load(self, path):
        """Load a trained form of this model from a path. Only supported
        for some subclasses.
        """

        raise NotImplementedError("This model does not support loading from "
                                  "files")

    def save(self, path):
        """Write this model to a path. Only supported for some
        subclasses.
        """

        raise NotImplementedError("This model does not support saving "
                                  "to files")

    def translate(self, word, n=5):
        """Translate the given word from source language to target language.

        Returns a list of target language words sorted by decreasing translation
        probability.
        """

        try:
            source_vec = self.source_vsm[word]
        except KeyError:
            raise ValueError(u"Word '{}' not found in source VSM".format(word))

        target_vec = self.translate_vec(source_vec)

        # TODO use KD tree for nearest-neighbor lookup
        ret = sorted(self.target_vsm.iterkeys(),
                     key=lambda v: distance.cosine(target_vec, v))
        return ret[:n]

    def translate_vec(self, source_vec):
        """Translate the word represented by the given word vector in
        source-language space to a vector in the target language space.
        """

        raise NotImplementedError("abstract method")
