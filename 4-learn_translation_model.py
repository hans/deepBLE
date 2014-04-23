import codecs
import logging
import pickle
import sys

from gensim.models import Word2Vec
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from scipy.spatial import distance


def learn(vsm_source, vsm_target, seeds, hidden_layer_size=100, bias=True,
          learning_rate=0.075):
    """Learn a translation model. Returns the learned neural network."""

    input_size = vsm_source.layer1_size
    output_size = vsm_target.layer1_size
    network = buildNetwork(input_size, hidden_layer_size, output_size,
                           bias=bias, fast=False)

    dataset = SupervisedDataSet(vsm_source.layer1_size, vsm_target.layer1_size)
    for source_word, target_word in seeds:
        try:
            source = vsm_source[source_word]
        except KeyError:
            logging.warn(u'Source VSM missing word {}'.format(source_word))
            continue

        try:
            target = vsm_target[target_word]
        except KeyError:
            logging.warn(u'Target VSM missing word {}'.format(target_word))
            continue

        dataset.addSample(source, target)

    trainer = BackpropTrainer(network, dataset, learningrate=learning_rate,
                              verbose=True)
    trainer.trainUntilConvergence()

    return network

def test_word(model, vsm_source, vsm_target, word, n=5,
              distfunc=distance.cosine):
    """Test the translation model by translating the given source-language
    word."""

    try:
        source_vec = vsm_source[word]
    except KeyError:
        logging.error(u'Source VSM missing word {}'.format(word))
        return

    translated = model.activate(source_vec)

    def target_word_score(target_word):
        return distfunc(translated, vsm_target[target_word])

    return sorted(vsm_target.vocab.iterkeys(),
                  key=target_word_score)[:n]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    vsm_source = Word2Vec.load(sys.argv[1])
    vsm_target = Word2Vec.load(sys.argv[2])

    with codecs.open(sys.argv[3], 'r', encoding='utf-8') as seeds_file:
        seeds = [tuple(line.strip().split('\t')) for line in seeds_file]
        print seeds

    model = learn(vsm_source, vsm_target, seeds)
    with open('translation-network', 'wb') as network_file:
        pickle.dump(model, network_file)

    print test_word(model, vsm_source, vsm_target, 'cat')
