from gensim.models import Word2Vec
from nose.tools import assert_equal

from model import NeuralTranslationModel
from model.runner import evaluate_model


def load_vsm_fixture(name):
    return Word2Vec.load('test/fixtures/{}'.format(name))


def test_3words():
    data = [('here', 'there'), ('now', 'then')]

    model = NeuralTranslationModel(
        load_vsm_fixture('three-words/vsm-a-window5-min0-dim5'),
        load_vsm_fixture('three-words/vsm-b-window5-min0-dim5'),
        hidden_layer_size=5, learning_rate=0.00001)
    model.train(data)

    print evaluate_model(model, data)
