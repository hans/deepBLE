from collections import defaultdict
import logging

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from deepble.model import TranslationModel
from deepble.model.linear import LinearTranslationModel


class ClusteredTranslationModel(TranslationModel):
    """Learns a mapping between clusters of the source VSM and the
    target VSM."""

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
        self.num_clusters = int(num_clusters) or 1

        # k * N array of cluster centroids (where k = number of
        # clusters). Constructed lazily in `build_clusters`
        self.clusters = None

        # Per-cluster submodels
        self.submodel = submodel
        self.models = None

    def build_clusters(self):
        """Retrieve all word representations from the source VSM and
        compute clusters"""

        self.clusters = MiniBatchKMeans(n_clusters=self.num_clusters)

        print 'Learning %i clusters' % self.num_clusters
        self.clusters.fit(self.source_vsm.syn0)

        # TODO print cluster information

    def train_vecs(self, source_vecs, target_vecs):
        if not self.clusters:
            self.build_clusters()

        self.models = [None] * self.num_clusters

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

        for cluster_id in range(self.num_clusters):
            model = self.submodel(self.source_vsm, self.target_vsm)
            source_vecs, target_vecs = zip(*(vecs_by_cluster[cluster_id]))
            model.train_vecs(list(source_vecs), list(target_vecs))

            self.models[cluster_id] = model

    def translate_vec(self, source_vec):
        if self.clusters is None or self.models is None:
            raise RuntimeError("Model not yet trained")

        cluster_id = self.clusters.predict([source_vec])[0]
        return self.models[cluster_id].translate_vec(source_vec)

    def load_object(self, obj):
        self.clusters, submodel_type, submodel_data = obj
        self.num_clusters = self.clusters.n_clusters

        self.models = []
        for submodel_data in submodel_data:
            model = submodel_type(self.source_vsm, self.target_vsm)
            model.load_object(submodel_data)

            self.models.append(model)

    def save_object(self):
        return (self.clusters, self.submodel,
                [model.save_object() for model in self.models])
