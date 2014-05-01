"""Evaluate specific translation models with given training and test
data.
"""

import argparse
import codecs
from datetime import date
import logging
from multiprocessing.pool import Pool
import random
import sys

from gensim.models import Word2Vec
from numpy import mean, std

import model


def train_test_split(data, training_pct=80):
    """Split input data into training and test sets."""

    random.shuffle(data)
    split = int(len(data) * (training_pct / 100.0))
    return data[:split], data[split:]


MODEL = None
def score((source_word, target_word), threshold=20):
    """Test the given model on the given `(source_word, target_word)`
    pair and return a score describing its performance.

    The score is bounded between 0 (worst) and 1 (perfect
    translation)."""

    try:
        predictions = MODEL.translate(source_word, n=threshold)
    except ValueError:
        # Translation error
        logging.exception("Translation error")
        return None

    try:
        rank = predictions.index(target_word)
    except ValueError:
        score = 0
    else:
        score = 1.0 / (rank + 1)

    return score


def evaluate_model(model, data, do_train=True):
    """Evaluate the performance of a translation model. Returns the mean
    and standard deviation of scores among data tuples used for testing
    (where a score ranges between 0 (worst) and 1 (perfect
    translation))."""

    if do_train:
        training_pairs, test_pairs = train_test_split(data)
        model.train(training_pairs)
    else:
        test_pairs = data

    global MODEL
    MODEL = model

    pool = Pool(4)
    scores = [x for x in pool.imap_unordered(score, test_pairs)
              if x is not None]
    logging.debug("Scores: %r" % scores)

    return mean(scores), std(scores)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the performance of a model.')

    parser.add_argument('model', choices=MODEL_MAPPING.keys())
    parser.add_argument('-f', '--model-file',
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

    return parser.parse_args()


MODEL_MAPPING = {
    'linear': model.LinearTranslationModel,
    'neural': model.NeuralTranslationModel,
}

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    # logging.getLogger().setFormatter(
    #     logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))

    arguments = parse_args()

    vsm_source = Word2Vec.load(arguments.vsm_source)
    vsm_target = Word2Vec.load(arguments.vsm_target)

    # Instantiate model
    model_class = MODEL_MAPPING[arguments.model]
    model = model_class(vsm_source, vsm_target)

    if arguments.model_file is not None:
        logging.debug("Loading model from file '{}'"
                      .format(arguments.model_file))

        try:
            model.load(arguments.model_file)
        except NotImplementedError:
            logging.error("Requested model does not support loading from "
                          "saved files")
            sys.exit(1)

    # Load seed data
    with codecs.open(arguments.data, 'r', encoding='utf-8') as data_file:
        data = [tuple(line.strip().split('\t')) for line in data_file]

    # Do we need to train the model?
    do_train = arguments.model_file is None

    # TODO print nicely
    print evaluate_model(model, data, do_train)

    if do_train:
        # We just trained a model -- save it somewhere
        model_name = "model-{}-{}".format(arguments.model, date.isoformat())
        model.save(model_name)
