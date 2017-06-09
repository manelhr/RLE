from rle.util.verbose_object import VerboseObject
from sklearn.model_selection import train_test_split


class Explanation(VerboseObject):
    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 model,
                 explainer, sampler, depicter,
                 num_samples=500,
                 verbose=False):
        """
        :param features: a np.array of np.arrays containing the different features.
        :param f_names: a list with the strings of the names features.
        :param f_types: a list with the data types of the features.
        :param label: a np.array containing a label.
        :param l_name: a string with the name of the label.
        :param l_type: the type of the label.
        :param model: np model for classification of the features.
        :param explainer: explainer object.
        :param sampler: sampler object.
        :param depicter: depicter object.
        :param num_samples: Number of samples sampled.
        :param verbose:
        """

        self.model = model
        self.explainer, self.sampler, self.depicter = explainer, sampler, depicter
        self.features_train, self.features_test, self.label_train, self.label_test = train_test_split(features, label)

        # Trains the model
        self._model = model()
        self._model.fit(self.features_train, self.label_train)
        self.classifier_fn = self.model.predict_proba

        # Initializes sampler
        self._sampler = sampler(features, f_names, f_types,
                                label, l_name, l_type, num_samples,
                                self.classifier_fn, verbose)

        super().__init__(verbose)
