from sklearn.linear_model import LogisticRegression
from rle.explainers.explainer import Explainer
from sklearn.metrics import accuracy_score
import pandas as pd
from functools import partial


class LogisticRegressionExplainer(Explainer):
    """ This class explains a decision with a weighted logistic regression, w/ a exponential kernel for weighting. """

    def __init__(self,
                 sampler):
        """ defined@Explainer This initializes the logistic regression.

        :param sampler: defined@Explainer
        :return defined@Explainer
        """

        self.model = LogisticRegression()

        self.sample_f, self.sample_l, self.last_decision = None, None, None

        self.pred_l = None

        self.exponential_sim, self.distances = None, None

        self.weights, self.metric, self.measure, self.num_samples = None, None, None, None

        super().__init__(sampler)

    def explain(self,
                decision,
                measure=None,
                num_samples=None):
        """ defined@Explainer We use a weighted logistic regression, where weights are given by a exponential kernel.

        :param decision: The point of interest to be explained.
        :param measure: If not none, replace the standard measure.
        :param num_samples: Number of samples to be used in the sampler.
        :return: Array with tuples ('feature', importance) where importance is a real number amd sum(importances) = 1.
        """

        self.measure = self.sampler.measure if measure is None else measure
        self.num_samples = self.sampler.num_samples if num_samples is None else num_samples
        self.last_decision = decision

        self.sample_f, self.sample_l, self.exponential_sim = self.sampler.sample(decision, num_samples, measure)

        self.model.fit(self.sample_f, self.sample_l, sample_weight=self.exponential_sim)

        self.weights = list(zip(self.sampler.feature_names(),
                                list(self.model.coef_)[0])) + \
                       [('Intercept', self.model.intercept_[0])]

        return self.weights

    def explanation_data(self):
        """ This provides the data on the explanation of the last explanation given.

        :return: Decision and dataframe with the samples/labels.
        """

        if self.sample_f is None or self.sample_l is None:
            raise Exception("Model not initialized.")

        df = pd.DataFrame(self.sample_f, columns=self.sampler.feature_names())

        df["Label"] = self.sample_l

        df["Exp. Sim."] = self.exponential_sim

        return self.last_decision, df

    def metrics(self, given_function=False):
        """ This provides the metrics on the model on the last sampled neighborhood.

        :param given_function: Possibly a function that receives y_true, y_pred and return some metric.
        :return: The accuracy score or custom metric.
        """

        self.pred_l = self.model.predict(self.sample_f)
        f = partial(accuracy_score, sample_weight=self.exponential_sim) if not given_function else given_function
        self.metric = f(self.sample_l, self.pred_l)

        return f(self.sample_l, self.pred_l)

    def explanation_result(self):
        """  defined@Explainer

        :return: Dictionary with weights and evaluation metric of the model.
        """

        if self.sample_f is None or self.sample_l is None:
            raise Exception("Model not initialized.")

        if self.weights is None:
            raise Exception("Model not fitted.")

        if self.metric is None:
            raise Exception("Metric not calculated.")

        return {"weights": self.weights,
                "metric": self.metric,
                "measure": self.measure,
                "num_sam": self.num_samples}
