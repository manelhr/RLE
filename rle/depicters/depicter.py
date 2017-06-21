import abc
from rle.util.verbose_object import VerboseObject


class Depicter(VerboseObject):
    """
    This is the abstract class that needs to be extended to implement the depicters for different models.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 destination=None,
                 verbose=False):
        """
        This initializes the depicter with its destination, if the destination is None plot will be shown on terminal.
        :param destination: possible destination for the object.
        :param verbose: defined@VerboseObject
        """

        self.destination = destination

        super().__init__(verbose)

    @abc.abstractmethod
    def depict(self, explanation_result):
        """
        This depicts an explanation result in any sort of way.
        :param explanation_result: dictionary with the aspects of the result of an explanation.
        :return: Nothing
        """

        pass
