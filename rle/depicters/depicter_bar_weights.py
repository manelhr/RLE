from rle.depicters.depicter import Depicter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DepicterBarWeights(Depicter):
    """ This abstract class depicts any explainer that has weights. """

    def __init__(self,
                 destination=None):
        """defined@Depicter"""

        super().__init__(destination)

    def depict(self, explanation_result, axis=None):
        """ Depicts explanation with weights as a bar chart.
        :param explanation_result: defined@Depicter
        :param axis: defined@Depicter
        :return: defined@Depicter
        """

        sns.set_style("whitegrid")
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        weights, labels, count, max_v = [], [], 0, 0

        for i in explanation_result['weights']:
            if i[0] == 'Intercept':
                continue
            labels.append(i[0])
            weights.append(i[1])
            count += 1
            max_v = max(max_v, abs(i[1]))

        # the bar centers on the y axis
        pos = np.arange(count) + .5

        if axis is None:
            fig, ax = plt.subplots(1, figsize=(4, 4))
        else:
            ax = axis

        ax.barh(pos, weights, color="green", alpha=0.6, align='center')
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Weights')
        ax.set_xlim([-max_v - 0.1 * max_v, max_v + 0.1 * max_v])
        ax.set_title("Acc:" + str(explanation_result['metric'])[:4] + " / " +
                     "$l$:" + str(explanation_result['measure'])[:4] + " / " +
                     "$n$:" + str(explanation_result['num_sam']))

        if axis is not None:
            return
        elif self.destination is not None:
            plt.savefig(self.destination, bbox_inches="tight")
        else:
            plt.show()
