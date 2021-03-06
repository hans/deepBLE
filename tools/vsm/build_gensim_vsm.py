import argparse
import logging
import multiprocessing

from gensim.corpora.dictionary import Dictionary

from deepble.vsm.corpora.text import TextCorpus
from deepble.vsm.corpora.wiki import WikiSentenceCorpus
from deepble.vsm.word2vec import Word2Vec


DEFAULT_WINDOW_SIZE = 10
DEFAULT_MINIMUM_TOKEN_COUNT = 200
DEFAULT_VECTOR_DIMENSIONS = 100


CORPUS_TYPES = {
    'plain': TextCorpus,
    'wiki': WikiSentenceCorpus
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a vector-space model from a text corpus.')

    parser.add_argument('corpus_path', help='Path to a text corpus')
    parser.add_argument('out_path',
                        help='Path to which to save the generated VSM')
    parser.add_argument('-t', '--corpus-type', choices=CORPUS_TYPES.keys(),
                        help='Format of the given corpus',
                        default='plain')

    word2vec_opts = parser.add_argument_group('Word2Vec options')
    word2vec_opts.add_argument('-w', '--window-size', type=int,
                               help='Word context window size',
                               default=DEFAULT_WINDOW_SIZE)
    word2vec_opts.add_argument('-m', '--minimum-token-count', type=int,
                               default=DEFAULT_MINIMUM_TOKEN_COUNT,
                               help=('Drop tokens which appear fewer '
                                     'times than this threshold in the '
                                     'corpus'))
    word2vec_opts.add_argument('-s', '--vector-dimensions', type=int,
                               default=DEFAULT_VECTOR_DIMENSIONS,
                               help=('Desired dimensionality of the '
                                     'vectors produced'))
    word2vec_opts.add_argument('--drop-capitals', action='store_true',
                               default=False,
                               help=('Ignore words in the vocabulary '
                                     'which begin with a capital '
                                     'letter'))

    dictionary_opts = parser.add_mutually_exclusive_group()
    dictionary_opts.add_argument('-d', '--dictionary-path',
                                 help=('Path to a pre-built dictionary '
                                       'corresponding to this corpus'))
    dictionary_opts.add_argument('--dictionary-out-path',
                                 help=('Path to which a computed '
                                       'dictionary should be saved.'))

    # TODO difference between dictionary and vocab? Do we need both?
    vocab_opts = parser.add_mutually_exclusive_group()
    vocab_opts.add_argument('--vocab-path',
                            help=('Path to a pre-built Word2Vec '
                                  'vocabulary corresponding to this '
                                  'corpus'))
    vocab_opts.add_argument('--vocab-out-path',
                            help=('Path to which a computed Word2Vec '
                                  'vocabulary should be saved'))

    parser.add_argument('--processed-corpus-save-path',
                        help=('Path to which to save the processed '
                              'corpus file (can later be reused)'))

    parser.add_argument('-v', '--verbose', action='store_true')

    return parser.parse_args()


def main(args):
    if args.corpus_type != 'wiki':
        if args.processed_corpus_save_path is not None:
            raise ValueError("Processed corpus saving only supported "
                             "for 'wiki' corpus type")

    kwargs = {}
    if args.dictionary_path is not None:
        kwargs['dictionary'] = Dictionary.load(args.dictionary_path)
    if args.dictionary_out_path is not None:
        kwargs['dictionary_save_path'] = args.dictionary_out_path

    if args.corpus_type == 'wiki' and args.processed_corpus_save_path is not None:
        kwargs['sentences_save_path'] = args.processed_corpus_save_path

    logging.debug('Building corpus')
    corpus = CORPUS_TYPES[args.corpus_type](args.corpus_path, **kwargs)
    documents = corpus.get_texts()

    logging.debug('Now beginning VSM construction with Word2Vec')

    model = Word2Vec(sentences=documents, vocab_path=args.vocab_path,
                     window=args.window_size, drop_capitals=args.drop_capitals,
                     min_count=args.minimum_token_count,
                     size=args.vector_dimensions,
                     workers=multiprocessing.cpu_count())

    model.save(args.out_path)

    if args.vocab_out_path is not None:
        model.save_vocab(args.vocab_out_path)


if __name__ == '__main__':
    arguments = parse_args()

    if arguments.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main(arguments)
