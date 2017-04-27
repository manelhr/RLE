import abc
from rle.samplers.sampler import Sampler


class ProximitySampler(Sampler):
    """ This is the abstract class that needs to be extended to implement the samplers for different models. """

    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 verbose):

        super().__init__(features, f_names, f_types,
                         label, l_name, l_type,
                         verbose)

    @abc.abstractmethod
    def sample(self,
               instance):

        pass

