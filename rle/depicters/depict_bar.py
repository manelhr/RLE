from rle.depicters.depicter import Depicter


class DepicterBar(Depicter):
    """
    This abstract class depicts any explainer that has weights.
    """

    def __init__(self,
                 destination=None,
                 verbose=False):
        """
        defined@Depicter
        """

        super().__init__(destination, verbose)

    def depict(self, explanation_result):
        """
        defined@Sampler This specific class depicts is as a bar chart.
        :param explanation_result: defined@Sampler
        :return: defined@Sampler
        """

        pass
