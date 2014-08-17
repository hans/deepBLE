import logging
import numpy as np
import pickle

import gensim.models
from gensim.models.word2vec import Vocab
from gensim.utils import smart_open


class Word2Vec(gensim.models.Word2Vec):
    """Defines a thin layer over the Gensim Word2Vec implementation:

    - Support serializing and loading vocabularies
        + It is recommended that you provide a full-coverage (i.e., unpruned)
          vocabulary, and use command-line options to describe how it may be
          pruned dynamically at runtime.
        + You can use the `min_count` argument to specify the minimum corpus
          term frequency in the dynamically pruned vocab, and the
          `drop_capitals` argument to indicate that terms beginning with
          capital letters should not be included in the final vocabulary.
    """

    def __init__(self, vocab_path=None, drop_capitals=False, **kwargs):
        self.drop_capitals = drop_capitals

        self.loaded_vocab = None
        if vocab_path is not None:
            logging.info('Loading Word2Vec vocabulary from {}'
                         .format(vocab_path))
            with open(vocab_path, 'r') as vocab_f:
                self.loaded_vocab = pickle.load(vocab_f)

        super(Word2Vec, self).__init__(**kwargs)

    @classmethod
    def load_glove_format(cls, vectors_path, vocab_path, norm_only=True):
        """
        Load a VSM model from the text format saved by the GloVe tool.

        See `load_word2vec_format` for description of `norm_only` param.
        """

        result = Word2Vec()

        logging.info("Loading vocab from {}".format(vocab_path))
        with smart_open(vocab_path, 'rb') as vocab_in:
            for index, line in enumerate(vocab_in):
                word, count = line.strip().split()
                result.vocab[word] = Vocab(index=index, count=count)

        logging.info("Loading projection weights from {}".format(vectors_path))

        first_run = True
        with smart_open(vectors_path, 'rb') as vectors_in:
            for index, line in enumerate(vectors_in):
                parts = line.split()

                # Set metadata by examining the first line's contents
                if first_run:
                    result.layer1_size = len(parts) - 1
                    result.syn0 = np.zeros(
                        (len(result.vocab), result.layer1_size),
                        dtype=np.float32)

                    first_run = False

                if len(parts) != result.layer1_size + 1:
                    raise ValueError("invalid vector on line {}".format(index))

                word, weights = parts[0], map(np.float32, parts[1:])

                result.index2word.append(word)
                result.syn0[index] = weights

        logging.info("Loaded {} matrix from {}"
                     .format(result.syn0.shape, vocab_path))

        result.init_sims(norm_only)
        return result

    def build_vocab(self, sentences):
        # Default to expensive way if we don't have a vocab loaded from file
        if self.loaded_vocab is None:
            super(Word2Vec, self).build_vocab(sentences)
            return

        self.vocab = self.loaded_vocab
        self.index2word = []

        to_delete = []

        # Start trimming the vocab by class `min_count` parameter, and
        # update `index2word` as we go
        i = 0
        for word, vocab_item in self.vocab.iteritems():
            if (vocab_item.count >= self.min_count
                and (not self.drop_capitals or not word.istitle())):
                vocab_item.index = i
                self.index2word.append(word)
                i += 1
            else:
                to_delete.append(word)

        for word in to_delete:
            del self.vocab[word]

        self.create_binary_tree()
        self.reset_weights()

    def save_vocab(self, vocab_path):
        """Save the built vocabulary to the given path.
        """

        if not self.vocab:
            raise RuntimeError("Attempted to save unbuilt vocab")

        with open(vocab_path, 'w') as vocab_f:
            pickle.dump(self.vocab, vocab_f)
