"""Evaluate specific translation models with given training and test
data.
"""

import argparse
import codecs
import logging

from gensim.models import Word2Vec
import gevent
from numpy import mean, std

import model


def train_test_split(data, training_pct=80):
    """Split input data into training and test sets."""

    split = int(len(data) * (training_pct / 100.0))
    return data[:split], data[split:]


def score(model, (source_word, target_word), threshold=5):
    """Test the given model on the given `(source_word, target_word)`
    pair and return a score describing its performance.

    The score is bounded between 0 (worst) and 1 (perfect
    translation)."""

    predictions = model.translate(source_word)

    try:
        rank = predictions.index(target_word)
    except ValueError:
        score = 0
    else:
        score = 1.0 / (rank + 1)

    return score


def evaluate_model(model_class, model_args, source_vsm, target_vsm, data):
    """Evaluate the performance of a translation model. Returns the mean
    and standard deviation of scores among data tuples used for testing
    (where a score ranges between 0 (worst) and 1 (perfect
    translation))."""

    training_pairs, test_pairs = train_test_split(data)

    model = model_class(source_vsm, target_vsm, **model_args)
    model.train(training_pairs)

    score_greenlets = [gevent.spawn(score, model, pair)
                       for pair in test_pairs]
    gevent.joinall(score_greenlets)

    scores = [greenlet.value for greenlet in score_greenlets]
    logging.debug("Scores: %r" % scores)

    return mean(scores), std(scores)


MODEL_MAPPING = {
    'linear': model.LinearTranslationModel,
    'neural': model.NeuralTranslationModel,
}

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description='Evaluate the performance of a model.')

    parser.add_argument('model', choices=MODEL_MAPPING.keys())
    parser.add_argument('-f', '--model-file', type=open,
                        help=('Saved model file (only supported for '
                              'some models)'))
    parser.add_argument('-s', '--vsm-source', required=True,
                        help=('Path to gensim word2vec file for source '
                              'language'))
    parser.add_argument('-t', '--vsm-target', required=True,
                        help=('Path to gensim word2vec file for target '
                              'language'))
    parser.add_argument('-d', '--data', required=True,
                        help=('Path to data TSV file (used for model '
                              'seeding (if training) and testing)'))

    arguments = parser.parse_args()

    vsm_source = Word2Vec.load(arguments.vsm_source)
    vsm_target = Word2Vec.load(arguments.vsm_target)

    # TODO handle model file parameter

    model_class = MODEL_MAPPING[arguments.model]
    with codecs.open(arguments.data, 'r', encoding='utf-8') as data_file:
        data = [tuple(line.strip().split('\t')) for line in data_file]

    # TODO print nicely
    print evaluate_model(model_class, {}, vsm_source, vsm_target, data)

    if arguments.model_file:
        arguments.model_file.close()
