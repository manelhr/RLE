from rle.util.inherit_docstring import inherit_docstring
from rle.util.verbose_object import VerboseObject
from rle.explanation.explanation import Explanation


@inherit_docstring
class RobustExplanation(VerboseObject):
    """ This class provides a robust explanation based on the neighborhood. """

    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 model,
                 explainer,
                 sampler,
                 depicter=None,
                 destination=None,
                 decision=None,
                 verbose=False):
        """ Initializes the robust explainer.
        :param features: defined@Explanation
        :param f_names: defined@Explanation
        :param f_types: defined@Explanation
        :param label: defined@Explanation
        :param l_name: defined@Explanation
        :param l_type: defined@Explanation
        :param model: defined@Explanation
        :param explainer: defined@Explanation
        :param sampler: defined@Explanation
        :param depicter: defined@Explanation
        :param decision: defined@Explanation
        :param verbose: defined@VerboseObject
        """

        self.explanation = Explanation(features=features, f_names=f_names, f_types=f_types,
                                       label=label, l_name=l_name, l_type=l_type,
                                       model=model,  explainer=explainer, sampler=sampler,
                                       depicter=depicter, destination=destination,
                                       decision=decision, verbose=verbose)


    def sample_explain_depict(self, decision, num_samples=None, measure=None, depict=True):
        """ This method samples the explanation calculates the metrics. If the depict parameter is not False, depicts.
        :param decision: decision of interest.
        :param num_samples: number of samples to be sampled by the sampler.
        :param measure: measure to be used in the model.
        :param depict: either a boolean (if false doesn't depict, if true depicts) or a custom depicter.
        :return: if depict is False, the explanation result, else, nothing.
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
                self._depicter.depict(self._exp_result)

        else:
            depict.depict(self._exp_result)