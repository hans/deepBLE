import argparse
from codecs import open
import logging
import re

from gensim.corpora import TextCorpus, WikiCorpus
from gensim.models import Word2Vec
from gensim.utils import tokenize


# TODO 3 builds better space than 5 -- do some more structured
# evaluation of window sizes
WINDOW_SIZE = 3
MINIMUM_TOKEN_COUNT = 5

SENTENCE_BOUNDARY = re.compile(r'\.(?!\d)')


# TODO can this just be a function?
class SentenceGen(object):
    """A generator which yields tokenized sentences from a corpus.

    The constructor accepts a single argument which is a corpus object.
    If this corpus is a `TextCorpus`, it is assumed that each "document"
    in the corpus is a single sentence. If it is a `WikiCorpus`, each
    document is sentence-tokenized by this generator.
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        if isinstance(self.corpus, TextCorpus):
            for sentence in self.corpus.get_texts():
                yield sentence
        elif isinstance(self.corpus, WikiCorpus):
            for document in self.corpus.get_texts():
                for sentence in re.split(SENTENCE_BOUNDARY, document):
                    yield tokenize(sentence, lowercase=True)


CORPUS_TYPES = {
    'plain': TextCorpus,
    'wiki': WikiCorpus
}


def main(corpus_path, corpus_type, out_path):
    logging.debug('Building corpus')
    corpus = CORPUS_TYPES[corpus_type](corpus_path)
    sentences = SentenceGen(corpus)

    logging.debug('Now beginning VSM construction with Word2Vec')
    model = Word2Vec(sentences, window=WINDOW_SIZE,
                     min_count=MINIMUM_TOKEN_COUNT, workers=4)
    model.save(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build a vector-space model from a text corpus.')

    parser.add_argument('corpus_path', help='Path to a text corpus')
    parser.add_argument('out_path',
                        help='Path to which to save the generated VSM')
    parser.add_argument('-t', '--type', choices=CORPUS_TYPES.keys(),
                        help='Format of the given corpus',
                        default='plain')
    parser.add_argument('-v', '--verbose', action='store_true')

    arguments = parser.parse_args()

    if arguments.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main(arguments.corpus_path, arguments.type, arguments.out_path)
