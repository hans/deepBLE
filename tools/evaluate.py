"""Evaluate specific translation models with given training and test
data.
"""

import argparse
import codecs
from datetime import datetime
import json
import logging
import multiprocessing
from multiprocessing.pool import Pool
import random
import sys

from gensim.models import Word2Vec
from numpy import mean, std

import model.all
from model.runner import evaluate_model, score


def train_test_split(data, training_pct=80):
    """Split input data into training and test sets."""

    random.shuffle(data)
    split = int(len(data) * (training_pct / 100.0))
    return data[:split], data[split:]


def save_model(model, originating_arguments):
    """Save a newly trained model along with information about the
    arguments that generated it.."""

    now = datetime.now()
    model_name = ("saved_models/model-{}-{}-{:02d}{:02d}"
                  .format(model.__class__.__name__, now.date().isoformat(),
                          now.hour, now.minute))

    try:
        model.save(model_name)
    except NotImplementedError:
        logging.info("Model does not support saving to files; "
                     "skipping save")
    else:
        # Save argument information as well
        with open('{}_arguments.json'.format(model_name), 'w') as arguments_f:
            json.dump(originating_arguments, arguments_f)


def parse_cmdline_kwarg(kwarg):
    param, value = kwarg.split('=')
    return param, float(value)


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
    parser.add_argument('-m', '--model-arguments', action='append',
                        type=parse_cmdline_kwarg,
                        help=('Float or integer keyword arguments to '
                              'pass to the model (of the form )'))
    parser.add_argument('--vsm-binary', default=False, action='store_true',
                        help=('Indicate that the provided VSMs are '
                              'binary word2vec forms (not gensim)'))
    parser.add_argument('--test-on-train', default=False, action='store_true',
                        help='Test on the same data used to train')

    arguments = parser.parse_args()

    if arguments.model_arguments is None:
        arguments.model_arguments = {}
    else:
        arguments.model_arguments = dict(arguments.model_arguments)

    return arguments


MODEL_MAPPING = {
    'identity': model.all.IdentityTranslationModel,
    'linear': model.all.LinearTranslationModel,
    'neural': model.all.NeuralTranslationModel,
    'percentile_frequency': model.all.PercentileFrequencyTranslationModel,
    'random': model.all.RandomTranslationModel,
    'clustered_linear': model.all.ClusteredLinearTranslationModel,
    'affine': model.all.AffineTranslationModel,
}

def main(arguments):
    if arguments.vsm_binary:
        vsm_source = Word2Vec.load_word2vec_format(arguments.vsm_source,
                                                   binary=True)
        vsm_target = Word2Vec.load_word2vec_format(arguments.vsm_target,
                                                   binary=True)
    else:
        vsm_source = Word2Vec.load(arguments.vsm_source)
        vsm_target = Word2Vec.load(arguments.vsm_target)

    # Instantiate model
    model_class = MODEL_MAPPING[arguments.model]
    model = model_class(vsm_source, vsm_target, **arguments.model_arguments)

    # Load seed data
    with codecs.open(arguments.data, 'r', encoding='utf-8') as data_file:
        data = [tuple(line.strip().split('\t')) for line in data_file]
    # By default, use all the data as test data
    test_pairs = data

    # Attempt to load model from file
    if arguments.model_file is not None:
        logging.debug("Loading model from file '{}'"
                      .format(arguments.model_file))

        try:
            model.load(arguments.model_file)
        except NotImplementedError:
            logging.error("Requested model does not support loading from "
                          "saved files")
            sys.exit(1)
    else:
        if arguments.test_on_train:
            training_pairs = test_pairs
        else:
            training_pairs, test_pairs = train_test_split(data)

        model.train(training_pairs)

        save_arguments = vars(arguments)
        save_arguments['extra'] = {
            'training_pairs': training_pairs
        }

        save_model(model, save_arguments)

    # Now perform evaluation
    #
    # TODO print nicely
    scores = list(evaluate_model(model, test_pairs))
    print mean(scores), std(scores)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    # logging.getLogger().setFormatter(
    #     logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))

    arguments = parse_args()
    main(arguments)
