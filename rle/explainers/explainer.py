import abc
from rle.util.verbose_object import VerboseObject


class Explainer(VerboseObject):
    """ This is the abstract class that needs to be extended to implement the explainers for different models. """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 verbose=False):

        super().__init__(verbose)

    @abc.abstractmethod
    def explain(self, sampler):

        pass


# import numpy as np
# from scipy.spatial.distance import euclidean
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
#
# from rle.explainers.rle_base import RLEBase
#
#
# class RLELinear(RLEBase):
#     """ This is an implementation of a local explainer that uses a generalized linear model for classification.
#     It uses an exponential kernel like the one used in the library LIME. """
#
#     def __init__(self,
#                  data,
#                  feature_names,
#                  label,
#                  label_name,
#                  #distance=None,
#                  regressor=None,
#                  verbose=False):
#         """ This function initializes the numeric regressor.
#         :param data: numpy 2d array (mixed).
#         :param feature_names: list of names of the columns of data.
#         :param label: numpy 1d array (categorical).
#         :param label_name: name of values of the label.
#         :param scaler: scaler implemented as in the sklearn library.
#         :param regressor: linear_model implemented as in the sklearn library.
#         :param feature_selection: function which takes data, label and feature names and return relevant features.
#         :param eps: steepness of the kernel used.
#         :param verbose: annoys the hell out of you.
#         """
#
#         # Initializes distance
#         #self.distance = distance if distance is not None else euclidean
#
#
#
#         # Initializes label to be predicted
#         self.label, self.label_name = label, label_name
#
#         # Initializes regressor
#         self.regressor = regressor if regressor is not None else LinearRegression()
#         self.verboselly_explain("Regressor successfully initialized")
#
#         # Initializes eps
#         self.eps = eps
#
#         super().__init__(verbose)
#
#     def get_neighborhood(self,
#                          instance):
#
#         # gets neighborhood
#         neighborhood = np.array([self.distance(data_point, instance) for data_point in self.data])
#
#         return neighborhood[neighborhood > self.eps]
#
#     def explain_instance(self,
#                          complex_predictor,
#                          instance):
#
#         neighborhood = self.get_neighborhood(instance)
#         number_instances = len(neighborhood)
#         data_pp = self.scaler.fit_transform(self.data[:, [self.features.index(selected) for selected in self.features]])
#
#         pred_complex = np.array([complex_predictor(neighborhood_instance) for neighborhood_instance in self.data])
#
#         predictions_regress = self.regressor.fit(data_pp, pred_complex)
#
#         return number_instances
