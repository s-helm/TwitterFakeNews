import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

from FeatureEngineering.FeatureSelector import get_label, get_feature_selection
from Utility.CSVUtils import load_data_from_CSV
from Utility.Util import list_diff


class SelectionMethods:
    F_VALUE = f_classif
    CHI2 = chi2
    MUTUAL = mutual_info_classif

    DEFAULT = CHI2

    @staticmethod
    def perform_selection(sel, X, y=None):
        """performs the feature selection.
        -X: data
        -sel: selector"""
        # Fit the Model
        if y is not None:
            sel.fit(X, y)
        else:
            sel.fit(X)

        features = sel.get_support(
            indices=True)  # returns an array of integers corresponding to nonremoved features
        features = [column for column in X[features]]  # Array of all nonremoved features' names

        # Format and Return
        sel = pd.DataFrame(sel.transform(X))
        sel.columns = features
        return sel

    @staticmethod
    def perform_selection_return_only_features(sel, X, y=None):
        """performs the feature selection.
        -X: data
        -sel: selector"""
        # Fit the Model
        if y is not None:
            sel.fit(X, y)
        else:
            sel.fit(X)

        features = sel.get_support(
            indices=True)  # returns an array of integers corresponding to nonremoved features
        features = [column for column in X[features]]  # Array of all nonremoved features' names

        # Format and Return
        return features


    @staticmethod
    def variance_threshold_selector(X, threshold=0):
        print("Filter with variance threshold of {}".format(threshold))
        sel = VarianceThreshold(threshold=threshold)
        return SelectionMethods.perform_selection(sel, X)


    @staticmethod
    def select_k_best_features(X, y, k, test=DEFAULT):
        """selects the k best features according to statistical test. Default: chi-squared"""
        print("Filter with k = {}".format(k))
        sel = SelectKBest(test, k)
        return SelectionMethods.perform_selection_return_only_features(sel, X, y)

    @staticmethod
    def select_k_best(X, y, k, test=MUTUAL):
        """selects the k best features according to statistical test. Default: mutual information"""
        print("Filter with k = {}".format(k))
        sel = SelectKBest(test, k)
        return SelectionMethods.perform_selection(sel, X, y)


    @staticmethod
    def find_best_feature_set(results):
        max_f1 = 1
        max_res = None
        for res in results:
            if res.f1 > max_f1:
                max_f1 = res.f1
                max_res = res
        return max_res


