import operator

from model import TranslationModel


class PercentileFrequencyTranslationModel(TranslationModel):
    """Attempts to match input words with words in the target language
    which have similar percentile corpus frequencies."""

    def __init__(self, source_vsm, target_vsm):
        super(PercentileFrequencyTranslationModel, self).__init__(source_vsm,
                                                                  target_vsm)

        self.source_distr = self.build_percentile_distribution(source_vsm)
        self.target_distr = self.build_percentile_distribution(target_vsm)

    def build_percentile_distribution(self, vsm):
        """Build a word/corpus-frequency-percentile distribution given
        a VSM."""

        # Map word to corpus frequency, sort by increasing frequency
        freqs = sorted(((word, vocab.count)
                        for word, vocab in vsm.vocab.iteritems()),
                       key=operator.itemgetter(1))

        vocab_size = float(len(vsm.vocab))
        percentiles = {w: i / vocab_size for i, (w, _) in enumerate(freqs)}

        return percentiles

    def train_vecs(self, source_vecs, target_vecs):
        pass

    def translate(self, word, n=5):
        try:
            source_percentile = self.source_distr[word]
        except KeyError:
            raise ValueError(u"Word '{}' not found in source VSM".format(word))

        target_words = sorted(self.target_distr,
                              key=lambda w: abs(self.target_distr[w]
                                                - source_percentile))
        return target_words[:n]
