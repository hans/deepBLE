from codecs import open
import sys

from gensim import corpora
from gensim.models import Word2Vec
from gensim.utils import tokenize

class SentenceGen(object):
    def __init__(self, dictionary, corpus_path):
        self.dictionary = dictionary
        self.corpus_path = corpus_path

    def __iter__(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield tokenize(line, lowercase=True)


def main(corpus_path, out_path):
    print '-- Beginning dictionary construction'
    with open(corpus_path, 'r') as f:
        dictionary = corpora.Dictionary(tokenize(line, lowercase=True)
                                        for line in f)
    print '-- Finished building dictionary'

    sentences = SentenceGen(dictionary, corpus_path)

    model = Word2Vec(sentences, min_count=5, workers=4)
    model.save(out_path)

    # print '-- Serializing MmCorpus'
    # corpora.MmCorpus.serialize(out_path, corpus)
    # print '-- Finished serializing MmCorpus'

    # print '-- Calculating TFIDF'
    # corpus_tfidf = models.TfidfModel(corpus)[corpus]

    # print '-- Running LSA'
    # lsa = lsimodel.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    # corpus_lsa = lsa[corpus_tfidf]

    # print '-- Serializing LSA'
    # corpora.MmCorpus.serialize(out_path + '.lsa', corpus_lsa)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
