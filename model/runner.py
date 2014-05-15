"""Handles the training and evaluation of new translation models.

- Trains models given a seed set and input vector spaces.
- Performs parallelized evaluation of models given a test set."""

import logging
import multiprocessing
from multiprocessing.pool import Pool


# Global reference to the trained model being evaluated, shared among
# processes
MODEL = None


def score((source_word, target_word), threshold=20):
    """Test the given model on the given `(source_word, target_word)`
    pair and return a score describing its performance.

    Allow the model to make `threshold` number of guesses before
    concluding that we have a score of 0.

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


def evaluate_model(model, test_data):
    """Evaluate the performance of a trained translation model. Returns
    the scores among data tuples provided for testing (where a score
    ranges between 0 (worst) and 1 (perfect translation))."""

    global MODEL
    MODEL = model

    # pool = Pool(multiprocessing.cpu_count())
    # scores = [x for x in pool.imap_unordered(score, test_data)
    #           if x is not None]
    scores = [score(x) for x in test_data]

    return scores
