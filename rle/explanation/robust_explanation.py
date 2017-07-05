from rle.explanation.explanation import Explanation
from peakutils.peak import indexes
import matplotlib.pyplot as plt
import numpy as np


class RobustExplanation:
    """ This class provides a robust explanation based on the neighborhood. """

    def __init__(self,
                 features, f_names, f_types,
                 label, l_name, l_type,
                 model,
                 explainer,
                 sampler,
                 depicter=None,
                 decision=None):
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
        """

        self.explanation = Explanation(features=features, f_names=f_names, f_types=f_types,
                                       label=label, l_name=l_name, l_type=l_type,
                                       model=model,  explainer=explainer, sampler=sampler,
                                       depicter=depicter, decision=decision)

    def sample_explain_depict(self, decision, n_exp,
                              num_samples=None, measure_min=0.05,
                              measure_max=2, number_eval=500, dest=None):
        """ This method provide sa robust explanation by finding accuracy (or metric) peaks between the measures.
        :param decision: Decision of interest.

        :param n_exp: Max number of explanations.
        :param num_samples: Number of samples to be sampled by the sampler.
        :param measure_min: Min measure to be used in the model.
        :param measure_max: Max measure to be used in the model.
        :param number_eval: Number of evaluations between measure_min and measure_max.
        :param dest: Destination to save the image.
        """

        linspace = np.linspace(measure_min, measure_max, number_eval)
        metric, weights = [], []

        for i in linspace:
            tmp = self.explanation.sample_explain_depict(decision, num_samples=num_samples, measure=i, depict=False)
            metric.append(tmp['metric'])

        metric = np.array(metric)

        min_len = len(metric) / 10
        peaks = indexes(metric, 0.6, min_len)
        print(peaks)
        n_exp = min(n_exp, len(peaks))

        fig, axs = plt.subplots(1, 1+n_exp, figsize=(3 + 3*n_exp, 2.5))

        for i, j in zip(peaks, np.arange(1, n_exp+1)):
            axs[0].plot(linspace[i], metric[i], "b*")
            self.explanation.sample_explain_depict(decision, num_samples=5000,
                                                   measure=linspace[i], depict=True, axis=axs[j])
        axs[0].plot(linspace, metric)
        axs[0].set_xlabel("$\ell$")
        axs[0].set_title("Weighted Accuracy")

        if dest is None:
            plt.show()
        else:
            plt.savefig(dest, bbox_inches="tight")
