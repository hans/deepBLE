import argparse
from codecs import open
import logging
import re

from gensim.corpora import TextCorpus
from gensim.models import Word2Vec

from corpora.wiki import WikiSentenceCorpus


# TODO 3 builds better space than 5 -- do some more structured
# evaluation of window sizes
WINDOW_SIZE = 10
MINIMUM_TOKEN_COUNT = 100

SENTENCE_BOUNDARY = re.compile(r'\.(?!\d)')


CORPUS_TYPES = {
    'plain': TextCorpus,
    'wiki': WikiSentenceCorpus
}


def main(corpus_path, corpus_type, out_path):
    logging.debug('Building corpus')
    corpus = CORPUS_TYPES[corpus_type](corpus_path)
    # sentences = SentenceGen(corpus)
    documents = corpus.get_texts()

    logging.debug('Now beginning VSM construction with Word2Vec')

    # TODO Word2Vec expects sentences, and we're giving it documents..
    # is this a problem?
    model = Word2Vec(documents, window=WINDOW_SIZE,
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
