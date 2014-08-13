from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal

from model.affine import AffineTranslationModel


def test_basic_affine():
    # This is a translation that can't be done only with a linear
    # transform. Can we handle it?
    train_in =  [[0, 0, 0], [1, 0, 1], [1, 1, 1]]
    train_out = [[1, 0, 1], [2, 0, 2], [2, 0, 2]]

    test_in = [10, 16, 10]
    test_out = [11, 0, 11]

    # TODO mock VSM objects
    model = AffineTranslationModel(None, None)

    model.train_vecs(train_in, train_out)

    assert_array_almost_equal(test_out, model.translate_vec(test_in))
