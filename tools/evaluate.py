"""Evaluate specific translation models with given training and test
data.
"""

import argparse
import codecs
from datetime import datetime
from functools import partial
import json
import logging
import sys

from gensim.models import Word2Vec
from numpy import mean, std

import deepble.model.all as models
from deepble.model.runner import evaluate_model


def load_seed_data(path):
    """Load seed data from the TSV file at the given path."""

    with codecs.open(path, 'r', encoding='utf-8') as data_file:
        return [tuple(line.strip().split('\t')) for line in data_file]


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
    parser.add_argument('--data-train',
                        help=('Path to data TSV file for training'))
    parser.add_argument('--data-test',
                        help=('Path to data TSV file for testing'))
    parser.add_argument('-m', '--model-config',
                        help=('Path to JSON dictionary file of keyword '
                              'arguments to pass to the model'))
    parser.add_argument('--vsm-binary', default=False, action='store_true',
                        help=('Indicate that the provided VSMs are '
                              'binary word2vec forms (not gensim)'))

    arguments = parser.parse_args()

    # Enforce argument invariants
    if not arguments.data_train and not arguments.data_test:
        logging.error('No data provided with --data-train or --data-test')
        sys.exit(1)
    elif arguments.model_file is not None and arguments.data_train is not None:
        logging.error('Cannot retrain provided model -- please omit '
                      '--data-train or --model-file')
        sys.exit(1)

    if arguments.model_config is None:
        arguments.model_arguments = {}
    else:
        with open(arguments.model_config, 'r') as config_f:
            arguments.model_arguments = json.load(arguments.model_arguments)

    return arguments


MODEL_MAPPING = {
    'identity': models.IdentityTranslationModel,
    'linear': models.LinearTranslationModel,
    'neural': models.mlp.NeuralTranslationModel,
    'linear_sgd': partial(models.mlp.MLPTranslationModel,
                          config_file='deepble/model/mlp/config/linear.yaml'),
    'percentile_frequency': models.PercentileFrequencyTranslationModel,
    'random': models.RandomTranslationModel,
    'clustered/linear': partial(models.ClusteredTranslationModel,
                                submodel=models.LinearTranslationModel),
    'clustered/affine': partial(models.ClusteredTranslationModel,
                                submodel=models.AffineTranslationModel),
    'affine': models.AffineTranslationModel,
}

def main(arguments):
    if arguments.vsm_binary:
        vsm_source = Word2Vec.load_word2vec_format(arguments.vsm_source,
                                                   binary=True, norm_only=True)
        vsm_target = Word2Vec.load_word2vec_format(arguments.vsm_target,
                                                   binary=True, norm_only=True)
    else:
        vsm_source = Word2Vec.load(arguments.vsm_source)
        vsm_target = Word2Vec.load(arguments.vsm_target)

        # Compute normalized word vectors and drop the old ones
        vsm_source.init_sims(replace=True)
        vsm_target.init_sims(replace=True)

    # Instantiate model
    model_class = MODEL_MAPPING[arguments.model]
    model = model_class(vsm_source, vsm_target, **arguments.model_arguments)

    # Load seed data
    data_train = (load_seed_data(arguments.data_train)
                  if arguments.data_train else None)
    data_test = (load_seed_data(arguments.data_test)
                 if arguments.data_test else None)

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

    # Train model
    if data_train is not None:
        model.train(data_train)

        save_arguments = vars(arguments)
        save_arguments['extra'] = {
            'training_pairs': data_train
        }

        save_model(model, save_arguments)

    # Test model
    if data_test is not None:
        scores = list(evaluate_model(model, data_test))
        print mean(scores), std(scores)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(formatter='%(asctime)s : %(levelname)s : %(message)s')

    arguments = parse_args()
    main(arguments)
