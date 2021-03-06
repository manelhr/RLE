from sklearn.model_selection import train_test_split


class Explanation:
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
                 decision=None):
        """ Initializes the explainer.

        :param features: A np.array of np.arrays containing the different features.
        :param f_names: A list with the strings of the names features.
        :param f_types: A list with the data types of the features.
        :param label: A np.array containing a label.
        :param l_name: A string with the name of the label.
        :param l_type: The type of the label.
        :param model: sklearn model for classification of the features (.fit and .predict_proba).
        :param explainer: Explainer object.
        :param sampler: Sampler object.
        :param depicter: Depicter object.
        :param num_samples: Number of samples sampled.
        :param measure: Measure for neighborhood.
        :param decision: Sets standard decision.
        """

        self.features_train, self.features_test, self.label_train, self.label_test = train_test_split(features, label)

        # Trains the model
        self._model = model
        self._model.fit(self.features_train, self.label_train)
        self.classifier_fn = self._model.predict_proba

        # Initializes sampler
        self._sampler = sampler(features, f_names, f_types,
                                label, l_name, l_type,
                                num_samples, measure,
                                self.classifier_fn)

        # Initializes explainer
        self._explainer = explainer(self._sampler)

        # Initializes standard depicter
        if depicter is not None:
            self._depicter = depicter(destination)
        else:
            self._depicter = None

        # Initializes the result of an explanation
        self._exp_result = None

        # Initializes sample decision
        self._decision = decision

    def sample_explain_depict(self, decision, num_samples=None, measure=None,  depict=True, axis=None):
        """ This method samples the explanation calculates the metrics. If the depict parameter is not False, depicts.

        :param decision: Decision of interest.
        :param num_samples: Number of samples to be sampled by the sampler.
        :param measure: Measure to be used in the model.
        :param depict: Either a boolean (if false doesn't depict, if true depicts) or a custom depicter.
        :param axis: matplotlib sub axis where the depiction should be plotted.
        :return: If depict is False, the explanation result, else, nothing.
        """

        # Sample explain
        self._explainer.explain(decision, measure, num_samples)

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
                self._depicter.depict(self._exp_result, axis)

        else:
            depict.depict(self._exp_result, axis)
