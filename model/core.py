import logging

from scipy.spatial import distance


def get_word_vector(vsm, word, alternate_encodings=None):
    """Retrieve the word vector for a given Unicode-encoded word from
    a VSM.

    Attempts to abstract away some encoding ugliness."""

    if alternate_encodings is None:
        alternate_encodings = ['utf-8', 'latin-1']

    try:
        return vsm[word]
    except KeyError:
        for encoding in alternate_encodings:
            try:
                return vsm[word.encode(encoding)]
            except UnicodeEncodeError: pass
            except KeyError: pass

        # Still here?
        return None


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

        source_vecs, target_vecs = [], []
        for source_word, target_word in seeds:
            source = get_word_vector(self.source_vsm, source_word)
            if source is None:
                logging.warn(u'Source VSM missing word {}'.format(source_word))
                continue

            target = get_word_vector(self.target_vsm, target_word)
            if target is None:
                logging.warn(u'Target VSM missing word {}'.format(target_word))
                continue

            source_vecs.append(source)
            target_vecs.append(target)

        return self.train_vecs(source_vecs, target_vecs)

    def train_vecs(self, source_vecs, target_vecs):
        """Train the model on a vector seed set, two lists where the
        `i`th element of `target_vecs` represents the translation of the
        `i`th element of `source_vecs`.
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

    def can_translate(self, target_word):
        """Determine if we can successfully translate to the given
        word. (If the target VSM doesn't contain the word, there is no
        possible way to get this right!)"""

        if isinstance(target_word, unicode):
            target_word = target_word.encode('utf-8')
        return target_word in self.target_vsm

    def translate(self, word, n=5):
        """Translate the given word from source language to target language.

        Returns a list of target language words encoded as Unicode
        strings and sorted by decreasing translation probability.
        """

        # VSM stores words as byte strings -- encode if we receive a
        # Unicode string
        if isinstance(word, unicode):
            word = word.encode('utf-8')

        try:
            source_vec = self.source_vsm[word]
        except KeyError:
            raise ValueError(u"Word '{}' not found in source VSM".format(word))

        target_vec = self.translate_vec(source_vec)

        # TODO use KD tree (or NearPy?) for nearest-neighbor lookup
        ret = sorted(self.target_vsm.vocab.iterkeys(),
                     key=lambda v: distance.cosine(target_vec,
                                              self.target_vsm[v]))
        ret = ret[:n]

        # Return list of Unicode strings
        ret = [word.decode('utf-8') for word in ret]

        return ret

    def translate_vec(self, source_vec):
        """Translate the word represented by the given word vector in
        source-language space to a vector in the target language space.
        """

        raise NotImplementedError("abstract method")
