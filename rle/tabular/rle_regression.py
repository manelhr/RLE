from sklearn.linear_model import LinearRegression
from rle.base.rle_base import RLEBase


class RLERegression(RLEBase):
    """ This is the base class for locally learning explainable models from data """

    def __init__(self,
                 data,
                 feature_names,
                 label,
                 label_name,
                 categorical_features,
                 regressor=None,
                 feature_selection=None,
                 eps=None,
                 verbose=False):
        """
        :param data: numpy 2d array (mixed).
        :param feature_name: list of names of the columns of data.
        :param label: numpy 1d array (numerical).
        :param label_name: name of the label.
        :param categorical_features: list of names of the categorical features.
        :param eps: neighborhood size.
        :param verbose: annoys the hell out of you.
        """

        used_features = feature_names if feature_selection is None else feature_selection(data, label, feature_names)
        used_regressor = regressor if regressor is not None else LinearRegression()


        super(verbose)

    def get_neighborhood(self,
                         eps):
        return

    def explain_instance(self,
                         **kwargs):
        return
