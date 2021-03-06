import abc


class Explainer:
    """ This is the abstract class that needs to be extended to implement the explainers for different models. """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 sampler):
        """ Initializes an explainer with a sampler and a measure of importance of the distance between a randomly
        sampled point and the decision we want to explain. This could be for instance, the steepness of a exponential
        kernel.

        :param sampler: Sampler object.
        :return: Nothing.
        """

        self.sampler = sampler

    @abc.abstractmethod
    def explain(self, decision, measure=None):
        """ Explain a decision using an explainer. Returns an array with the importance of each feature.
        :param decision: The point of interest to be explained.
        :param measure: If not none, replace the standard measure.
        :return: Array with tuples ('feature', importance) where importance is a real number amd sum(importances) = 1.
        """

        pass

    @abc.abstractmethod
    def explanation_result(self):
        """ This function returns the explanation result to be used by the depicted.
        :return: Explanation result.
        """

        pass
