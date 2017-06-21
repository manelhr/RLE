import abc
from rle.util.verbose_object import VerboseObject


class Depicter(VerboseObject):
    """
    This is the abstract class that needs to be extended to implement the depicters for different models.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 verbose=False):

        super().__init__(verbose)

    @abc.abstractmethod
    def depict(self, explainer):

        pass
