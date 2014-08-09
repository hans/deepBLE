#!/usr/bin/env python

# First step in generating train/test data in the style of Mikolov et
# al. [1]:
#
# 1. Extract the top 6,000 words in a Word2Vec vocabulary
# 2. Find their top translations via Google Translate
# 3. Use the top 5K translation pairs as training seeds, and the
#    remaining 1K as test seeds. (Seeds are output as a single stream
#    sorted by corpus frequency -- they should be split into different
#    files for proper training/testing.)
#
# [1] Mikolov, Tomas, Quoc V. Le, and Ilya Sutskever. "Exploiting
#     similarities among languages for machine translation." arXiv
#     preprint arXiv:1309.4168 (2013).

from argparse import ArgumentParser

from gensim.models import Word2Vec


def get_top_words(vsm, n=6000, omit_stopwords=False):
    """Retrieve the top `n` words sorted by descending corpus
    frequency."""

    ret = sorted(vsm.vocab.iterkeys(), key=lambda k: vsm.vocab[k].count,
                 reverse=True)[:n]
    return [x.decode('utf-8') for x in ret]


def parse_args():
    parser = ArgumentParser(
        description=('Generate train/test data in the style of Mikolov '
                     'et al. (2013). See the source of this file for '
                     'more information.'))

    parser.add_argument('-v', '--vsm-path', required=True,
                        help='Path to a word2vec VSM (binary format)')

    return parser.parse_args()


def main(args):
    vsm = Word2Vec.load_word2vec_format(args.vsm_path, binary=True)
    words = get_top_words(vsm)

    print '\n'.join(x.encode('utf-8') for x in words)


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
