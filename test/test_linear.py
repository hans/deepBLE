from gensim.models import Word2Vec
from nose.tools import assert_equal

from model.linear import LinearTranslationModel
from model.runner import evaluate_model


def load_vsm_fixture(name):
    return Word2Vec.load('test/fixtures/{}'.format(name))


def test_3words():
    data = [('here', 'there'), ('now', 'then')]

    model = LinearTranslationModel(
        load_vsm_fixture('three-words/vsm-a-window5-min0-dim5'),
        load_vsm_fixture('three-words/vsm-b-window5-min0-dim5'))
    model.train(data)

    for score in evaluate_model(model, data):
        assert_equal(1.0, score, "Should see 100% performance on training data")
