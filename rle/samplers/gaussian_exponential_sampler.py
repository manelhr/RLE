from rle.util.inherit_docstring import inherit_docstring
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from rle.samplers.sampler import Sampler
import numpy as np


@inherit_docstring
class GaussianExponentialSampler(Sampler):
    """ This class implements a sampler based on generating a gaussian distribution around the point to be explained.
        Restrictions:
        1) Can only be used with numerical features.
    """

    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 num_samples, measure,
                 classifier_fn,
                 verbose=False):
        """ defined@Sampler Also initializes distances metrics.

        :param features: defined@Sampler
        :param f_names: defined@Sampler
        :param f_types: defined@Sampler
        :param label: defined@Sampler
        :param l_name: defined@Sampler
        :param l_type: defined@Sampler
        :param num_samples: defined@Sampler
        :param measure: defined@Sampler
        :param classifier_fn: defined@Sampler
        :param verbose: defined@VerboseObject
        :return: defined@Sampler
        """
        super().__init__(features, f_names, f_types,
                         label, l_name, l_type,
                         num_samples, measure,
                         classifier_fn,
                         verbose)

        self.scaler = StandardScaler(with_mean=False)

        self.scaler.fit(features)

    def sample(self,
               instance,
               num_samples=None,
               measure=None):
        """ defined@Sampler

        :param instance: defined@Sampler
        :param num_samples: defined@Sampler
        :param measure: defined@Sampler
        :return: defined@Sampler
        """

        ns = int(num_samples) if num_samples is not None else self.num_samples
        ms = measure if measure is not None else self.measure

        self.num_samples = ns

        s_features = np.random.normal(0, 1, ns * instance.shape[0])

        s_features = s_features.reshape(ns, instance.shape[0]) * self.scaler.scale_ + instance

        s_labels = np.array([np.argmax(inst) for inst in self.classifier_fn(s_features)])

        distances = pairwise_distances(s_features, instance.reshape(1, -1), metric='euclidean').ravel()

        exponential_sim = np.sqrt(np.exp(-(distances ** 2) / ms ** 2))

        return s_features, s_labels, exponential_sim

