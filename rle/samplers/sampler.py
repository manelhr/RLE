import abc
from rle.util.verbose_object import VerboseObject


class Sampler(VerboseObject):
    """ This is the abstract class that needs to be extended to implement the samplers for different models. """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 verbose=False):

        # initializes feature related variables
        self.features, self.f_names, self.f_types = features, f_names, f_types

        # initializes label related variables
        self.label, self.l_name, self.l_type = label, l_name, l_type

        super().__init__(verbose)

    @abc.abstractmethod
    def sample(self, instance):

        pass
