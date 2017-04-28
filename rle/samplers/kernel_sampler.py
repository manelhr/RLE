from scipy.spatial.distance import euclidean
from rle.util.inherit_docstring import inherit_docstring
from rle.samplers.sampler import Sampler


@inherit_docstring
class KernelSampler(Sampler):
    """ This class implements a sampler based on distance measurements. """

    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 kernel_fn, kernel_w, num_samples,
                 verbose=False):
        """
        defined@Sampler Also initializes distances metrics.

        :param features: defined@Sampler
        :param f_names: defined@Sampler
        :param f_types: defined@Sampler
        :param label: defined@Sampler
        :param l_name: defined@Sampler
        :param l_type: defined@Sampler
        :param kernel_fn: Function that transforms an array of distances into an  array of proximity values (floats).
        :param kernel_w: Weight used in the kernel.
        :param num_samples: Number of samples sampled.
        :param verbose: defined@VerboseObject
        :return: defined@Sampler
        """

        self.kernel_fn, self.kernel_w, self.num_samples = kernel_fn, kernel_w, num_samples

        super().__init__(features, f_names, f_types,
                         label, l_name, l_type,
                         verbose)

    def sample(self,
               instance):
        """
        defined@Sampler

        :param instance: defined@Sampler
        :return: defined@Sampler
        """



        return 0

    def test(self):
        return self

print(KernelSampler.sample.__doc__)