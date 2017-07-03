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
                 num_samples=500,
                 measure=1,
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
        :param num_samples: defined@Explanation
        :param measure: defined@Explanation
        :param decision: defined@Explanation
        :param verbose: defined@VerboseObject
        """

        self.explanation = Explanation(features, f_names, f_types,
                                       label, l_name, l_type,
                                       model,  explainer, sampler,
                                       depicter, destination,
                                       num_samples, measure, decision, verbose)
