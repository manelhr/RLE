import abc
from rle.util.verbose_object import VerboseObject


class LogisticRegressionExplainer(VerboseObject):
    """ This is the abstract class that needs to be extended to implement the explainers for different models. """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 verbose=False):

        super().__init__(verbose)

    @abc.abstractmethod
    def explain(self, sampler):

        pass
