#!/usr/bin/env python

# Generate train/test data in the style of Mikolov et al. [1]:
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
import json
import urllib

from gensim.models import Word2Vec


def get_top_words(vsm, n=6000, omit_stopwords=False):
    """Retrieve the top `n` words sorted by descending corpus
    frequency."""

    return sorted(vsm.vocab.iteritems(), key=lambda k: vsm.vocab[k].count,
                  reverse=True)[:n]


def chunk(seq, size):
    """Partition a sequence into chunks."""
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


TRANSLATION_URL = ("https://translate.google.com/translate_a/single?"
                   "client=t&sl={source}&tl={target}&hl=en&dt=bd&dt=ex"
                   "&dt=ld&dt=md&dt=qc&dt=rw&dt=rm&dt=ss&dt=t&dt=at"
                   "&dt=sw&ie=UTF-8&oe=UTF-8&pc=9&oc=1&otf=2&ssel=0"
                   "&tsel=0&q={input}")

def get_translations(words, source_language, target_language):
    """Get a single translation of each of the given `words` from source
    language `source_language` to target language `target_language` (ISO
    639-2 language codes)."""

    # Fetch translations in batches
    batch_size = 100

    translations = []
    for word_chunk in chunk(words, batch_size):
        word_input = urllib.quote(u'\n'.join(words))
        url = TRANSLATION_URL.format(source=source_language,
                                     target=target_language,
                                     input=word_input)

        response = json.loads(urllib.urlopen(url).read())
        translations.extend([word_candidates[0].strip()
                             for word_candidates in response[0][0]])

    return translations


def parse_args():
    parser = ArgumentParser(
        description=('Generate train/test data in the style of Mikolov '
                     'et al. (2013). See the source of this file for '
                     'more information.'))

    parser.add_argument('-v', '--vsm-path',
                        help='Path to a word2vec VSM (binary format)')
    parser.add_argument('-s', '--source',
                        help='Source language ISO 639-2 code')
    parser.add_argument('-t', '--target',
                        help='Target language ISO 639-2 code')

    return parser.parse_args()


def main(args):
    vsm = Word2Vec.load_word2vec_format(args.vsm_path, binary=True)
    words = get_top_words(vsm)
    translations = get_translations(words, args.source, args.target)

    for word, translation in zip(words, translations):
        print u'{}\t{}'.format(word.decode('utf-8'),
                               translation.decode('utf-8'))
