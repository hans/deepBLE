"""
Defines utility functions and classes for common operations in the
deepBLE code.
"""


class MockVSM(object):
    """
    A class which quacks like a `gensim.models.word2vec.Word2Vec`
    object, but is really just a collection of vectors.
    """

    def __init__(self, vectors):
        """
        Initialize with a list of vectors which are in this VSM.
        """

        # TODO need to store vectors?
        # Will upgrade this mock as necessary

        if not vectors:
            return

        self.layer1_size = len(vectors[0])
