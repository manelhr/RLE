from rle.util.inherit_docstring import inherit_docstring
from sklearn.preprocessing import StandardScaler
from rle.samplers.sampler import Sampler
import numpy as np


@inherit_docstring
class GaussianSampler(Sampler):
    """
    This class implements a sampler based on generating a gaussian distribution around the point to be explained.
        Restrictions:
        1) Needs classifier prediction probability function.
        2) Can only be used with numerical features.
    """

    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 num_samples, classifier_fn,
                 verbose=False):
        """
        defined@Sampler Also initializes distances metrics.

        :param features: defined@Sampler
        :param f_names: defined@Sampler
        :param f_types: defined@Sampler
        :param label: defined@Sampler
        :param l_name: defined@Sampler
        :param l_type: defined@Sampler
        :param num_samples: defined@Sampler
        :param classifier_fn: defined@Sampler
        :param verbose: defined@VerboseObject
        :return: defined@Sampler
        """
        super().__init__(features, f_names, f_types,
                         label, l_name, l_type,
                         num_samples, classifier_fn,
                         verbose)

        self.scaler = StandardScaler(with_mean=False)
        self.scaler.fit(features)

    def sample(self,
               instance,
               num_samples=None):
        """
        defined@Sampler

        :param instance: defined@Sampler
        :param num_samples: defined@Sampler
        :return: defined@Sampler
        """

        ns = num_samples if num_samples is not None else self.num_samples

        s_features = np.random.normal(0, 1, ns * instance.shape[0]) \
            .reshape(ns, instance.shape[0]) * self.scaler.scale_ + instance

        s_labels = np.array([np.argmax(inst) for inst in self.classifier_fn(s_features)])

        return s_features, s_labels

