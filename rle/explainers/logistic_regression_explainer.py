from rle.explainers.explainer import Explainer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


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

        self.sample_f, self.sample_l, self.last_decision = None, None, None

        self.pred_l = None

        self.exponential_sim, self.distances, self.explanation_result = None, None, None

        super().__init__(sampler, measure, verbose)

    def explain(self,
                decision,
                measure=None,
                num_samples=None):
        """
        defined@Explainer We use a weighted logistic regression, where the weights are given by a exponential kernel.

        :param decision: the point of interest to be explained.
        :param measure: if not none, replace the standard measure.
        :return: array with tuples ('feature', importance) where importance is a real number amd sum(importances) = 1.
        """

        ms = measure if measure is not None else self.measure

        self.last_decision = decision

        self.sample_f, self.sample_l = self.sampler.sample(decision, num_samples)

        self.distances = pairwise_distances(self.sample_f, decision.reshape(1, -1), metric='euclidean').ravel()

        self.exponential_sim = np.sqrt(np.exp(-(self.distances ** 2) / ms ** 2))

        self.model.fit(self.sample_f, self.sample_l, sample_weight=self.exponential_sim)

        self.explanation_result = list(zip(self.sampler.feature_names(),
                                           list(self.model.coef_)[0])) + \
                                  [('Intercept', self.model.intercept_[0])]

        return self.explanation_result

    def explanation_data(self):
        """
        This provides the data on the explanation of the last explanation given.
        :return: decision and dataframe with the samples/labels.
        """

        if self.sample_f is None or self.sample_l is None:
            raise Exception("Model not initialized.")

        df = pd.DataFrame(self.sample_f, columns=self.sampler.feature_names())

        df["Label"] = self.sample_l

        df["Distances"] = self.distances

        df["Exp. Sim."] = self.exponential_sim

        return self.last_decision, df

    def metrics(self, given_function=False):
        """
        This provides the metrics on the model on the last sampled neighborhood.
        :param given_function: possibly a function that receives y_true, y_pred and return some metric.
        :return: the accuracy score or custom metric.
        """

        self.pred_l = self.model.predict(self.sample_f)
        f = accuracy_score if not given_function else given_function

        return f(self.sample_l, self.pred_l)
