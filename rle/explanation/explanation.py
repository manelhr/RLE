from rle.util.verbose_object import VerboseObject
from sklearn.model_selection import train_test_split


class Explanation(VerboseObject):
    """ This class does everything: samples, explains and depicts. """

    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 model,
                 explainer,
                 sampler,
                 depicter=None,
                 destination=None,
                 num_samples=500,
                 measure=1,
                 verbose=False):
        """ Initializes the explainer.
        :param features: a np.array of np.arrays containing the different features.
        :param f_names: a list with the strings of the names features.
        :param f_types: a list with the data types of the features.
        :param label: a np.array containing a label.
        :param l_name: a string with the name of the label.
        :param l_type: the type of the label.
        :param model: sklearn model for classification of the features (.fit and .predict_proba).
        :param explainer: explainer object.
        :param sampler: sampler object.
        :param depicter: depicter object.
        :param num_samples: Number of samples sampled.
        :param verbose: defined@VerboseObject
        """

        self.features_train, self.features_test, self.label_train, self.label_test = train_test_split(features, label)

        # Trains the model
        self._model = model
        self._model.fit(self.features_train, self.label_train)
        self.classifier_fn = self._model.predict_proba

        # Initializes sampler
        self._sampler = sampler(features, f_names, f_types,
                                label, l_name, l_type, num_samples,
                                self.classifier_fn, verbose)

        # Initializes explainer
        self._explainer = explainer(self._sampler, measure, verbose)

        # Initializes standard depicter
        if depicter is not None:
            self._depicter = depicter(destination)
        else:
            self._depicter = None

        # Initializes the result of an explanation
        self._exp_result = None

        super().__init__(verbose)

    def sample_explain_depict(self, decision, num_samples=None, measure=None,  depict=True):
        """ This method samples the explanation calculates the metrics. If the depict parameter is not False, depicts.
        :param decision: decision of interest.
        :param num_samples: number of samples to be sampled by the sampler.
        :param measure: measure to be used in the model.
        :param depict: either a boolean (if false doesn't depict, if true depicts) or a custom depicter.
        :return: if depict is False, the explanation result, else, nothing.
        """

        # Sample explain
        self._explainer.explain(decision, num_samples, measure)

        # Calculate metrics
        self._explainer.metrics()

        # Get explanation result
        self._exp_result = self._explainer.explanation_result()

        if depict is False:
            return self._exp_result

        elif depict is True:
            if self._depicter is None:
                raise Exception("No depicter given.")
            else:
                self._depicter.depict(self._exp_result)

        else:
            depict.depict(self._exp_result)
