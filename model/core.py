import logging
import pickle

import numpy as np
from numpy.linalg import norm


class TranslationModel(object):
    """A model which uses paired vector-space language models to translate
    words between two languages.
    """

    def __init__(self, source_vsm, target_vsm):
        self.source_vsm = source_vsm
        self.target_vsm = target_vsm

    def _get_vsm_vec(self, vsm, word):
        if isinstance(word, unicode):
            word = word.encode('utf-8')

        vocab_item = vsm.vocab[word]
        return vsm.syn0norm[vocab_item.index]

    def _get_source_vec(self, word):
        """Get the vector representation of the given source-language
        word in the source VSM.
        """

        return self._get_vsm_vec(self.source_vsm, word)

    def _get_target_vec(self, word):
        """Get the vector representation of the given target-language
        word in the target VSM.
        """

        return self._get_vsm_vec(self.target_vsm, word)

    def train(self, seeds):
        """Train the model on a seed set, a list of pairs of the form

            (source_word, translationally_equivalent_target_word)
        """

        source_vecs, target_vecs = [], []
        for source_word, target_word in seeds:
            try:
                source = self._get_source_vec(source_word)
            except KeyError:
                logging.warn(u'Source VSM missing word {}'.format(source_word))
                continue

            try:
                target = self._get_target_vec(target_word)
            except KeyError:
                logging.warn(u'Target VSM missing word {}'.format(target_word))
                continue

            source_vecs.append(source)
            target_vecs.append(target)

        return self.train_vecs(source_vecs, target_vecs)

    def train_vecs(self, source_vecs, target_vecs):
        """Train the model on a vector seed set, two lists where the
        `i`th element of `target_vecs` represents the translation of the
        `i`th element of `source_vecs`.

        The provided vectors should be unit vectors.
        """

        raise NotImplementedError("abstract method")

    def load(self, path):
        """Load a trained form of this model from a path. Only supported
        for some subclasses.
        """

        logging.info("Loading {} from '{}'".format(type(self).__name__, path))
        with open(path, 'r') as f:
            return self.load_object(pickle.load(f))

    def load_object(self, obj):
        """Initialize the model from its serialized form. The provided
        parameter is the un-pickled serialized data of the model."""

        raise NotImplementedError("This model does not support loading from "
                                  "files")

    def save(self, path):
        """Write this model to a path. Only supported for some
        subclasses.
        """

        logging.info("Saving {} to '{}'".format(type(self).__name__, path))
        with open(path, 'w') as f:
            pickle.dump(self.save_object(), f)

    def save_object(self):
        """Provide an object with which to serialize the class. This
        same object type will be provided to the model in `load_object`.
        """

        raise NotImplementedError("This model does not support saving "
                                  "to files")

    def can_translate(self, target_word):
        """Determine if we can successfully translate to the given
        word. (If the target VSM doesn't contain the word, there is no
        possible way to get this right!)"""

        if isinstance(target_word, unicode):
            target_word = target_word.encode('utf-8')
        return target_word in self.target_vsm.vocab

    def translate(self, word, n=5):
        """Translate the given word from source language to target language.

        Returns a list of target language words encoded as Unicode
        strings and sorted by decreasing translation probability.
        """

        try:
            source_vec = self._get_source_vec(word)
        except KeyError:
            raise ValueError(u"Word '{}' not found in source VSM".format(word))

        target_vec = self.translate_vec(source_vec)

        # Ensure unit vector
        target_vec = target_vec / norm(target_vec)

        # TODO use KD tree (or NearPy?) for nearest-neighbor lookup
        ret = sorted(self.target_vsm.vocab.iterkeys(),
                     key=lambda v: np.dot(target_vec,
                                          self._get_target_vec(v)))
        ret = ret[:n]

        # Return list of Unicode strings
        ret = [word.decode('utf-8') for word in ret]

        return ret

    def translate_vec(self, source_vec):
        """Translate the word represented by the given word vector in
        source-language space to a vector in the target language space.
        """

        raise NotImplementedError("abstract method")
