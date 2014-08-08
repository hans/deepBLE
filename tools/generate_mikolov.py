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
import itertools
import json
import logging
import sys
import time
import urllib

from gensim.models import Word2Vec
import requests


def get_top_words(vsm, n=6000, omit_stopwords=False):
    """Retrieve the top `n` words sorted by descending corpus
    frequency."""

    ret = sorted(vsm.vocab.iterkeys(), key=lambda k: vsm.vocab[k].count,
                 reverse=True)[:n]
    return [x.decode('utf-8') for x in ret]


TRANSLATION_URL = ("http://glosbe.com/gapi_v0_1/translate?format=json"
                   "&from={source}&dest={target}&phrase={input}")

TRANSLATION_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/37.0.2062.68 Safari/537.36')
}

# Sleep between requests to avoid rate limiting
SLEEP_DURATION = 1

def get_translations(words, source_language, target_language):
    """Get a single translation of each of the given `words` from source
    language `source_language` to target language `target_language` (ISO
    639-3 language codes)."""

    for word in words:
        # Skip sentence boundary markers
        if word == "</s>":
            continue

        word_input = urllib.quote(word.encode('utf-8'))

        url = TRANSLATION_URL.format(source=source_language,
                                     target=target_language,
                                     input=word_input)

        response = requests.get(url, headers=TRANSLATION_HEADERS)

        data = response.json()
        candidates = data['tuc']
        if not candidates:
            logging.warn(u'No translations for source word "{}"'.format(word))
            continue

        candidate = candidates[0]
        if 'phrase' not in candidate:
            logging.warn(u'No full translation for source word "{}"'.format(word))
            continue

        yield word, candidate['phrase']['text']

        # Sleep between requests to avoid rate limiting
        time.sleep(SLEEP_DURATION)


def parse_args():
    parser = ArgumentParser(
        description=('Generate train/test data in the style of Mikolov '
                     'et al. (2013). See the source of this file for '
                     'more information.'))

    parser.add_argument('-v', '--vsm-path', required=True,
                        help='Path to a word2vec VSM (binary format)')
    parser.add_argument('-s', '--source', required=True,
                        help='Source language ISO 639-3 code')
    parser.add_argument('-t', '--target', required=True,
                        help='Target language ISO 639-3 code')

    return parser.parse_args()


def main(args):
    vsm = Word2Vec.load_word2vec_format(args.vsm_path, binary=True)
    words = get_top_words(vsm)
    translations = get_translations(words, args.source, args.target)

    for word, translation in translations:
        print '{}\t{}'.format(word.encode('utf-8'),
                              translation.encode('utf-8'))


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
