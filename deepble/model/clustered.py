from collections import defaultdict
import logging

import numpy as np
from sklearn import manifold
from sklearn.cluster import MiniBatchKMeans

from deepble.model import TranslationModel
from deepble.model.linear import LinearTranslationModel
from deepble.util import MockVSM


class ClusteredTranslationModel(TranslationModel):
    """
    Learns a mapping between clusters of the source VSM and the
    target VSM.

    For each cluster of the source VSM, we compute lower-dimensional
    representations of the words in the cluster. We then learn a mapping
    from this low-dimensional representation to the output space.
    """

    def __init__(self, source_vsm, target_vsm, submodel=LinearTranslationModel,
                 num_clusters=None):
        """Initialize the model and build a clustering of the source
        vector space.

        `submodel` is the model class which should be used to map
        between individual clusters. Any other model could theoretically
        be used here, though this class was designed for use with
        `LinearTranslationModel` and `AffineTranslationModel` as its
        submodels.

        If `num_clusters` is not provided at training time, we will
        try multiple cluster counts and pick the one with the lowest
        resulting distortion."""

        super(ClusteredTranslationModel, self).__init__(source_vsm,
                                                        target_vsm)

        # TODO guess clustering more scientifically
        self.num_clusters = (num_clusters and int(num_clusters)) or 500

        # k * N array of cluster centroids (where k = number of
        # clusters). Constructed lazily in `build_clusters`
        self.clusters = None

        # k-element array (where k = number of clusters) of transformers
        # which project source-VSM vectors onto the low-dimensional
        # embeddings particular to each cluster
        self.cluster_spaces = None

        # Per-cluster submodels
        self.submodel = submodel
        self.models = None

    def build_clusters(self):
        """
        Retrieve all word representations from the source VSM and
        compute clusters
        """

        self.clusters = MiniBatchKMeans(n_clusters=self.num_clusters,
                                        batch_size=int(self.num_clusters * 1.8))

        logging.info('Learning %i clusters', self.num_clusters)
        self.clusters.fit(self.source_vsm.syn0norm)

        # TODO print cluster information

    def build_cluster_transformer(self, source_vecs,
                                  method=manifold.LocallyLinearEmbedding):
        """
        Compute a lower-dimensional representation of the given
        source-VSM vectors, and return an object which can
        `.transform()` vectors from the source VSM into this
        lower-dimensional embedding.
        """

        # TODO try different dimensionalities
        dimensionality = 50

        transformer = method(n_components=dimensionality)
        transformer.fit(np.array(source_vecs))

        return transformer

    def train_vecs(self, source_vecs, target_vecs):
        if not self.clusters:
            self.build_clusters()

        self.models = [None] * self.num_clusters
        self.cluster_spaces = [None] * self.num_clusters

        # n_vecs * d array
        X_cluster = np.array(source_vecs)

        # array of cluster IDs corresponding to `source_vecs` elements
        cluster_ids = self.clusters.predict(X_cluster)

        vecs_by_cluster = defaultdict(list)
        for cluster_id, source_vec, target_vec in zip(cluster_ids, source_vecs,
                                                   target_vecs):
            vecs_by_cluster[cluster_id].append((source_vec, target_vec))

        for cluster_id, pairs in vecs_by_cluster.items():
            logging.info('Cluster %i has %i seeds', cluster_id, len(pairs))

        if set(vecs_by_cluster.keys()) != set(range(self.num_clusters)):
            # TODO more informative error
            raise RuntimeError("Missing source vectors for some "
                               "clusters")

        for cluster_id, vectors in vecs_by_cluster.iteritems():
            # source_vecs is a list of vectors in the global source VSM
            # -- not the cluster
            source_vecs, target_vecs = zip(*vectors)

            cluster_transformer = self.build_cluster_transformer(source_vecs)
            cluster_contents = cluster_transformer.predict(source_vecs)

            cluster_vsm = MockVSM(cluster_contents)

            model = self.submodel(cluster_vsm, self.target_vsm)
            model.train_vecs(list(cluster_contents), list(target_vecs))

            self.models[cluster_id] = model
            self.cluster_spaces[cluster_id] = cluster_transformer

    def translate_vec(self, source_vec):
        if self.clusters is None or self.models is None:
            raise RuntimeError("Model not yet trained")

        cluster_id = self.clusters.predict([source_vec])[0]

        # Project the vector into the low-dimensional space associated
        # with the matched cluster
        projected = self.cluster_spaces[cluster_id].predict([source_vec])[0]

        # Now translate this projected vector into the target space
        # using the submodel associated with the matched cluster
        return self.models[cluster_id].translate_vec(projected)

    def load_object(self, obj):
        self.clusters, self.cluster_spaces, submodel_type, submodel_data = obj
        self.num_clusters = self.clusters.n_clusters

        self.models = []
        for submodel_data in submodel_data:
            model = submodel_type(self.source_vsm, self.target_vsm)
            model.load_object(submodel_data)

            self.models.append(model)

    def save_object(self):
        return (self.clusters, self.cluster_spaces, self.submodel,
                [model.save_object() for model in self.models])
