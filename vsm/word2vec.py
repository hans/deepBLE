import logging
import pickle

import gensim.models


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
