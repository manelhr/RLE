import abc
from rle.util.verbose_object import VerboseObject


class Sampler(VerboseObject):
    """
    This is the abstract class that needs to be extended to implement the samplers for different models.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 num_samples, classifier_fn,
                 verbose=False):
        """
        Initializes a sampler with attributes related to a dataset's features and its label. Only Work for datasets with
        only one label.

        :param features: a np.array of np.arrays containing the different features.
        :param f_names: a list with the strings of the names features.
        :param f_types: a list with the data types of the features.
        :param label: a np.array containing a label.
        :param l_name: a string with the name of the label.
        :param l_type: the type of the label.
        :param num_samples: Number of samples sampled.
        :param classifier_fn: classifier prediction probability function, which takes a numpy array and outputs
        prediction probabilities. For ScikitClassifiers, this is classifier.predict_proba.
        :param verbose: defined@VerboseObject.
        :return: Nothing.
        """

        # initializes feature related variables
        self.features, self.f_names, self.f_types = features, f_names, f_types

        # initializes label related variables
        self.label, self.l_name, self.l_type = label, l_name, l_type

        self.num_samples = num_samples
        self.classifier_fn = classifier_fn

        super().__init__(verbose)

    @abc.abstractmethod
    def sample(self, instance):
        """
        This method creates samples to an instance, based on the desired method of sampling.

        :param instance: numpy array with instance to be sampled.
        :return: a tuple (features, label) of the sample.
        """
        pass
