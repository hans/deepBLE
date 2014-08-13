from mock import Mock
from numpy.testing import assert_array_almost_equal

from model.clustered import ClusteredTranslationModel
from model.linear import LinearTranslationModel


def test_clustered():
    """Basic test to make sure we properly learn separate mappings."""

    # Train with two clusters which have different linear
    # transformations

    train_in =  [[1., 1.], [0., 1.], [3., 8. ]]
    train_out = [[2., 2.], [0., 2.], [9., 24.]]

    test_in = [1., 1.1]
    test_out = [2., 2.2]

    source_vsm = Mock()
    source_vsm.syn0 = train_in
    target_vsm = Mock()
    target_vsm.syn0 = train_out

    model = ClusteredTranslationModel(source_vsm, target_vsm, num_clusters=2,
                                      submodel=LinearTranslationModel)

    model.train_vecs(train_in, train_out)

    assert_array_almost_equal(test_out, model.translate_vec(test_in))
