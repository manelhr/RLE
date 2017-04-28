from scipy.spatial.distance import euclidean
from rle.util.inherit_docstring import inherit_docstring
from rle.samplers.sampler import Sampler


@inherit_docstring
class ProximitySampler(Sampler):
    """ This class implements a sampler based on distance measurements. """

    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 dist_num, dist_cat, dist_mix,
                 verbose=False):
        """
        defined@Sampler Also.

        :param features: defined@Sampler
        :param f_names: defined@Sampler
        :param f_types: defined@Sampler
        :param label: defined@Sampler
        :param l_name: defined@Sampler
        :param l_type: defined@Sampler
        :param dist_num: distance metric between numerical features.
        :param dist_cat: distance metric between categorical features.
        :param dist_mix: distance metric between a numerical feature and a categorical feature.
        :param verbose: defined@VerboseObject
        :return: defined@Sampler
        """
        super().__init__(features, f_names, f_types,
                         label, l_name, l_type,
                         verbose)


    def sample(self,
               instance):
        return 0

    def test(self):
        return self

print(ProximitySampler.__init__.__doc__)