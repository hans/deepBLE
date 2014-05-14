from nose.tools import assert_equal

from tools import evaluate


class DummyNamespace(object):
    """Fake an argparse Namespace result"""

    def __init__(self, entries):
        self.__dict__.update(entries)


def test_evaluate_dummy_neural():
    arguments = {
        'model': 'neural',
        'vsm_source': 'test/fixtures/three-words/vsm-a-window5-min0-dim5',
        'vsm_target': 'test/fixtures/three-words/vsm-b-window5-min0-dim5',
        'data': 'test/fixtures/three-words/seed-set.tsv',
        'model_arguments': {'hidden_layer_size': 5},
        'model_file': None,
        'test_on_train': True,
    }

    evaluate.main(DummyNamespace(arguments))
