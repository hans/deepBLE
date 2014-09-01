from itertools import izip

import numpy as np
from sklearn.linear_model import LogisticRegression

from deepble.model.core import TranslationModel


class LogisticRegressionTranslationModel(TranslationModel):
    """A translation model which trains a logistic-regression classifier
    on the given training examples.
    """

    C = 1.0
    """Inverse of regularization strength for logistic regression
    classifier. Smaller floats indicate stronger regularization.
    """

    def __init__(self, *args, **kwargs):
        super(LogisticRegressionTranslationModel, self).__init__(*args, **kwargs)

        self.model = None

    def build_training_data(self, source_vecs, target_vecs):
        """Build a classifier training data pair `(X, y)`, where `X` is
        a matrix of examples (one example per row) and `y` carries
        classifier labels for the corresponding examples in `X`.
        """

        # Build positive examples
        X_pos = np.mat([np.concatenate((source, target))
                        for source, target in izip(source_vecs, target_vecs)])
        y_pos = np.repeat(1, X_pos.shape[0])

        # Sample random negative examples by pairing non-translations

        # TODO: Make as many negative examples as positive examples. Is
        # this the optimal amount?
        num_negative = X_pos.shape[0]

        negative_indices = [(i, j)
                            for i in range(len(source_vecs))
                            for j in range(len(source_vecs))
                            if i != j]
        negative_example_indices = np.random.choice(negative_indices,
                                                    num_negative)
        X_neg = np.mat([np.concatenate((source_vecs[i], target_vecs[j]))
                        for i, j in negative_example_indices])
        y_neg = np.repeat(0, X_neg.shape[0])

        # Construct final result
        X, y = np.concatenate((X_pos, X_neg)), np.concatenate((y_pos, y_neg))

        return X, y

    def train_vecs(self, source_vecs, target_vecs):
        # Learning task:
        #
        # - Input: Concatenation of source-language word vector and
        #   target-language word vector
        # - Output: Probability of the two relevant words being
        #   translations of one another

        X, y = self.build_training_data(source_vecs, target_vecs)

        self.model = LogisticRegression(C=self.C)
        self.model.fit(X, y)

    def load_object(self, obj):
        self.model = obj

    def save_object(self):
        return self.model

    def translate(self, word, n=5):
        try:
            source_vec = self._get_source_vec(word)
        except KeyError:
            raise ValueError(u"Word '{}' not found in source VSM".format(word))

        # Make a prediction using every vector in the target vocabulary
        # concatenated to this source vector. Return the top predictions
        # in decreasing sorted order
        def score(target_word):
            # Classifier input
            x = np.concatenate(source_vec, self._get_target_vec(target_word))

            return self.model.predict(x)

        ret = sorted(self.target_vsm.vocab.iterkeys(), key=score, reverse=True)
        return ret[:n]

    def translate_vec(self, source_vec):
        raise NotImplementedError("Model does not support word vector "
                                  "output")
