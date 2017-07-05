import abc


class Depicter:
    """ This is the abstract class that needs to be extended to implement the depicters for different models. """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 destination=None):
        """ Initializes the depicter with its destination, if the destination is None plot will be shown on terminal.
        :param destination: Possible destination for the object.
        """

        self.destination = destination

    @abc.abstractmethod
    def depict(self, explanation_result, axis):
        """ This depicts an explanation result in any sort of way.
        :param explanation_result: Dictionary with the aspects of the result of an explanation.
        :param axis: matplotlib sub axis where the depiction should be plotted.
        :return: Nothing.
        """

        pass
