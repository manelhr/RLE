from rle.explainers.explainer import Explainer
from sklearn.linear_model import LogisticRegression


class LogisticRegressionExplainer(Explainer):
    """
    This is the abstract class that needs to be extended to implement the explainers for different models.
    """

    def __init__(self,
                 sampler,
                 measure,
                 verbose=False):
        """
        :param sampler: defined@Explainer
        :param measure: defined@Explainer
        :param verbose: defined@Explainer
        :return defined@Explainer
        """

        super().__init__(sampler, measure, verbose)

    def explain(self, sampler):

        pass
