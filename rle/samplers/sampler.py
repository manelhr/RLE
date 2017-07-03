import abc
from rle.util.verbose_object import VerboseObject


class Sampler(VerboseObject):
    """ This is the abstract class that needs to be extended to implement the samplers for different models. """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 num_samples, measure,
                 classifier_fn,
                 verbose=False):
        """ Initializes a sampler with attributes related to a dataset's features and its label. Only Work for datasets
        with only one label.

        :param features: A np.array of np.arrays containing the different features.
        :param f_names: A list with the strings of the names features.
        :param f_types: A list with the data types of the features.
        :param label: A np.array containing a label.
        :param l_name: A string with the name of the label.
        :param l_type: The type of the label.
        :param num_samples: Number of samples sampled by default.
        :param measure: Measure of locality to be used by the sampler.
        :param classifier_fn: Classifier prediction probability function, which takes a numpy array and outputs
        prediction probabilities. For ScikitClassifiers, this is classifier.predict_proba.
        :param verbose: defined@VerboseObject
        :return: nothing.
        """
        self.features, self.f_names, self.f_types = features, f_names, f_types  # initializes feature related variables
        self.label, self.l_name, self.l_type = label, l_name, l_type  # initializes label related variables
        self.num_samples = int(num_samples)
        self.measure = measure
        self.classifier_fn = classifier_fn

        super().__init__(verbose)

    @abc.abstractmethod
    def sample(self, instance, num_samples=None, measure=None):
        """ This method creates samples to an instance, based on the desired method of sampling.

        :param instance: Numpy array with instance to be sampled.
        :param num_samples: Number of samples. If not None, replace the standard number of samples.
        :param measure: Measure of locality. If not None, replace the standard measure of locality.
        :return: A tuple (features, label) of the sample.
        """
        pass

    def feature_names(self):
        """ Getter for the feature names.

        :return: f_names array.
        """
        return self.f_names
