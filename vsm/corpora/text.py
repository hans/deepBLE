import logging

import gensim.corpora
from gensim.corpora.dictionary import Dictionary


class TextCorpus(gensim.corpora.TextCorpus):
    """A corpus class which makes some minor extensions to the Gensim
    `TextCorpus` implementation:

    - Support loading of pre-built dictionary
    """

    def __init__(self, input=None, dictionary=None, dictionary_save_path=None):
        super(gensim.corpora.TextCorpus, self).__init__()

        self.input = input
        self.metadata = False

        if dictionary is None:
            self.dictionary = Dictionary()

            if input is not None:
                self.dictionary.add_documents(self.get_texts())
            else:
                logging.warning("No input document stream provided; "
                                "assuming dictionary will be "
                                "initialized in some other way.")
        else:
            self.dictionary = dictionary

        if dictionary_save_path is not None:
            self.dictionary.save(dictionary_save_path)
