from gensim.models import Word2Vec
from nose.tools import assert_equal

from model.neural import NeuralTranslationModel
from model.runner import evaluate_model


def load_vsm_fixture(name):
    m = Word2Vec.load('test/fixtures/{}'.format(name))
    m.init_sims()
    return m


def test_3words():
    data = [('here', 'there'), ('now', 'then')]

    model = NeuralTranslationModel(
        load_vsm_fixture('three-words/vsm-a-window5-min0-dim5'),
        load_vsm_fixture('three-words/vsm-b-window5-min0-dim5'),
        hidden_layer_size=5, learning_rate=0.01)
    model.train(data)

    # TODO how to assert good enough performance here? Maybe train an ensemble?
    print evaluate_model(model, data)
