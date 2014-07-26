import logging

import gensim.corpora
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import getstream
import gensim.utils

class TextCorpus(gensim.corpora.TextCorpus):
    """A corpus class which makes some minor extensions to the Gensim
    `TextCorpus` implementation:

    - Support loading of pre-built dictionary
    """

    def __init__(self, input=None, dictionary=None, dictionary_save_path=None,
                 pre_tokenized=False, lowercase=False):
        super(gensim.corpora.TextCorpus, self).__init__()

        self.input = input
        self.metadata = False

        self.pre_tokenized = pre_tokenized
        self.lowercase = lowercase

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

    def get_texts(self):
        length = 0

        # Input should have one document (sentence, for the word2vec case) per line
        for line in getstream(self.input):
            length += 1

            if self.pre_tokenized:
		if not isinstance(line, unicode):
		    line = unicode(line, encoding='utf8', errors='strict')
                yield line
            else:
                yield gensim.utils.tokenize(line, lowercase=self.lowercase)

        self.length = length
