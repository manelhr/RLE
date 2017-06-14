import abc
from rle.util.verbose_object import VerboseObject


class Explainer(VerboseObject):
    """
    This is the abstract class that needs to be extended to implement the explainers for different models.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 sampler,
                 measure,
                 verbose=False):
        """
        Initializes an explainer with a sampler and a measure of importance of the distance between a randomly sampled
        point and the decision we want to explain. This could be for instance, the steepness of a exponential kernel.

        :param sampler: sampler object.
        :param measure: measure of importance of the distance between sampled point and the decision .
        :param verbose: defined@VerboseObject
        :return: nothing.
        """

        self.sampler = sampler

        self.measure = measure

        super().__init__(verbose)

    @abc.abstractmethod
    def explain(self, decision, measure=None):
        """
        Explain a decision using an explainer. Returns an array with the importance of each feature.
        :param decision: the point of interest to be explained.
        :param measure: if not none, replace the standard measure.
        :return: array with tuples ('feature', importance) where importance is a real number amd sum(importances) = 1.
        """

        pass
