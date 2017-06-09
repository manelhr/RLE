from rle.explainers.explainer import Explainer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


class LogisticRegressionExplainer(Explainer):
    """
    This class explains a decision with a weighted logistic regression, using a exponential kernel for weighting.
    """

    def __init__(self,
                 sampler,
                 measure,
                 verbose=False):
        """
        defined@Explainer This initializes the logistic regression.

        :param sampler: defined@Explainer
        :param measure: defined@Explainer
        :param verbose: defined@Explainer
        :return defined@Explainer
        """

        self.model = LogisticRegression()

        super().__init__(sampler, measure, verbose)

    def explain(self, decision, measure=None):
        """
        defined@Explainer We use a weighted logistic regression, where the weights are given by a exponential kernel.

        :param decision: the point of interest to be explained.
        :param measure: if not none, replace the standard measure.
        :return: array with tuples ('feature', importance) where importance is a real number amd sum(importances) = 1.
        """

        ms = measure if measure is not None else self.measure

        sample_f, sample_l = self.sampler.sample()

        distances = pairwise_distances(sample_f, decision.reshape(1, -1), metric='euclidean').ravel()
        exponential_distances = np.sqrt(np.exp(-(distances ** 2) / ms ** 2))
        self.model.fit(sample_f, sample_l, exponential_distances)
