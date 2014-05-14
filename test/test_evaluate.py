from nose.tools import assert_equal

from tools import evaluate


class DummyNamespace(object):
    """Fake an argparse Namespace result"""

    def __init__(self, entries):
        self.__dict__.update(entries)


def test_evaluate_dummy_neural():
    arguments = {
        'model': 'neural',
        'vsm_source': 'test/fixtures/single-word/vsm-a-window5-min0-dim2',
        'vsm_target': 'test/fixtures/single-word/vsm-b-window5-min0-dim2',
        'data': 'test/fixtures/single-word/seed-set.tsv',
        'model_arguments': {'hidden_layer_size': 2},
        'model_file': None,
    }

    evaluate.main(DummyNamespace(arguments))
