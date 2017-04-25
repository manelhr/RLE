import abc


class RLEBase(object):
    """ This is the base class for locally learning explainable models from data """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 verbose=False):
        self.verbose = verbose

    @abc.abstractmethod
    def get_neighborhood(self,
                         **kwargs):
        return

    @abc.abstractmethod
    def explain_instance(self,
                         **kwargs):
        return

    def verboselly_explain(self,
                           string):
        print(string) if self.verbose else None
