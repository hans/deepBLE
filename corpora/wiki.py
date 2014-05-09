"""Defines a corpus class for dealing with Wikipedia corpus data."""

import bz2
import codecs
import logging
import multiprocessing
import os.path
import re

from gensim.corpora.wikicorpus import (ARTICLE_MIN_WORDS, WikiCorpus,
                                       _extract_pages, filter_wiki, tokenize)
from gensim import utils


LOGGER = logging.getLogger('corpora.wiki')

# SENTENCE_BOUNDARY = re.compile(r"""
#     (?<=\S)                        # some word material
#     [.!?]                          # potential sentence ending
#     (?=(?P<after_tok>
#         (?:[?!)\";}\]\*:@\'\({\[]) # other punctuation
#         |
#         \s+(?P<next_tok>\S+)       # or whitespace and some other token
#     ))""", re.UNICODE | re.VERBOSE)
# """Regular expression which matches sentence boundaries, extracted from
# NLTK's [`PunktSentenceTokenizer`][1].

# [1]:http://www.nltk.org/_modules/nltk/tokenize/punkt.html#PunktSentenceTokenizer
# """

# SENTENCE_BOUNDARY = re.compile(r"""[.!?](?:['"]\s)?(?![,\d])""")

# Slightly improved: doesn't split on e.g., i.e.
#
# Probably hitting diminishing returns with this sentence segmentation
# stuff.
SENTENCE_BOUNDARY = re.compile(r"""(?<!\s)[.!?](?:['"]\s)?(?![,\d])""")


def process_article(args):
    """Parse a Wikipedia article, returning its content as a list of
    sentences (each a list of utf8-encoded token strings).
    """

    text, do_lemmatize, title, pageid = args
    text = filter_wiki(text)

    process_fn = utils.lemmatize if do_lemmatize else tokenize
    sentences = []
    for sentence in re.split(SENTENCE_BOUNDARY, text):
        sentences.append(process_fn(sentence))

    return sentences, title, pageid


class WikiSentenceCorpus(WikiCorpus):
    """
    Extends the built-in Gensim `WikiCorpus` to yield *sentences* rather
    than entire *documents* at a time. This corpus then matches the
    specifications of the `Word2Vec` Gensim model.

    Several other notable extensions have been made:

    - Support reading from uncompressed Wikipedia dump
    """

    def __init__(self, *args, **kwargs):
        super(WikiSentenceCorpus, self).__init__(*args, **kwargs)

    def open_corpus_file(self):
        _, extension = os.path.splitext(self.fname)
        if extension == 'bz2':
            return bz2.BZ2File(self.fname)
        else:
            return codecs.open(self.fname, encoding='utf-8')

    def get_texts(self):
        """
        Iterate over the corpus data, yielding sentences of the text
        version of each article (each sentence represented as a list of
        tokens).

        See the `WikiCorpus` class for more.
        """

        # Unfortunately due to the OOP-unfriendly implementation of
        # `WikiCorpus` we have to copy-and-paste some code. This code is
        # based on `WikiCorpus#get_texts`.

        n_articles, n_articles_all = 0, 0
        n_sentences, n_sentences_all = 0, 0

        pages = _extract_pages(self.open_corpus_file, self.filter_namespaces)
        texts = ((text, self.lemmatize, title, pageid)
                 for title, text, pageid in pages)

        pool = multiprocessing.Pool(self.processes)

        # process the corpus in smaller chunks of docs, because
        # multiprocessing.Pool is dumb and would load the entire input
        # into RAM at once...
        chunks = utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1)

        for group in chunks:
            for sentences, title, pageid in pool.imap(process_article, group):
                n_articles_all += 1
                n_sentences_all += len(sentences)

                num_tokens = sum(len(sentence) for sentence in sentences)

                # article redirects and short stubs are pruned here
                if num_tokens > ARTICLE_MIN_WORDS:
                    n_articles += 1
                    n_sentences += len(sentences)

                    for sentence in sentences:
                        if self.metadata:
                            yield (sentence, (pageid, title))
                        else:
                            yield sentence

        pool.terminate()

        LOGGER.info("finished iterating over Wikipedia corpus of %i "
                    "articles with %i sentences (%i articles / %i "
                    "sentences retained)" %
                    (n_articles_all, n_sentences_all, n_articles, n_sentences))

        # cache corpus length
        self.length = n_sentences
