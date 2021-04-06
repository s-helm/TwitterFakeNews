import json
import copy
from pathlib import Path
from threading import Thread

import gc
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from DatasetUtils.Standardizer import standardize_data
from FeatureEngineering.FeatureSelector import get_label, get_group_feature
from Learning.EvaluationMetrics import calc_accuracy, calc_f1, calc_recall, calc_precision


class XValidation:
    """class that is responsible for evaluation"""

    def __init__(self, features):
        self.conf_matr = np.zeros((2, 2))
        self.conf_matr_list = list()
        self.features = features

    def groupKFold(self, data, clf, n_splits=10, standardize=False):
        """performs a n_splits X Validation where tweets of the
        same user are never in training and testing at the same time"""
        model_name = type(clf).__name__
        print("Perform {}-fold X-Validation with {}".format(n_splits, model_name))

        label = get_label()
        groups = get_group_feature()

        X = data[self.features].as_matrix()
        y = data[label].as_matrix()
        groups = data[groups].as_matrix()

        print(X.shape)

        self.conf_matr = np.zeros((2, 2))

        threads_coll = []
        gkf = GroupKFold(n_splits)
        i = 0

        for train, test in gkf.split(X, y, groups=groups):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            if standardize and (isinstance(clf, GaussianNB) or
                                isinstance(clf, MLPClassifier) or
                                isinstance(clf, LinearSVC) or
                                isinstance(clf, Pipeline)):
                print("Standardize Data")
                X_train, X_test = standardize_data(X_train, X_test)

            clf_i = copy.deepcopy(clf)

            t = Thread(target=self.train_and_predict, args=(clf_i, X_train, X_test, y_train, y_test, i))
            threads_coll.append(t)
            i += 1
            nr_of_cores = 1

            if i % nr_of_cores == 0 or i == n_splits:
                # Start all threads
                for x in threads_coll:
                    x.start()

                # Wait for all of them to finish
                for x in threads_coll:
                    x.join()
                    print("Thread {} done.".format(x))
                threads_coll = list()
            gc.collect()

        for conf in self.conf_matr_list:
            self.conf_matr = np.add(self.conf_matr, conf)

    def train_and_predict(self, model, X_train, X_test, y_train, y_test, i=None):
        """trains a single model"""

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if type(model) == XGBClassifier:
            y_pred = [round(value) for value in y_pred]

        self.conf_matr_list.append(np.array(confusion_matrix(y_test, y_pred, [1, 0])))

        del X_train
        del X_test
        del y_train
        del y_test

    def get_accuracy(self):
        return calc_accuracy(self.conf_matr)

    def get_precision(self):
        return calc_precision(self.conf_matr)

    def get_recall(self):
        return calc_recall(self.conf_matr)

    def get_F1_score(self):
        return calc_f1(self.conf_matr)

    def save_fold_indices(self, train, test):
        """
        saves training and test indices
        :param config: 
        :param filename: 
        :return: 
        """
        filename = 'x_val_folds.json'
        my_file = Path(filename)
        if my_file.is_file():
            with open(filename) as file:
                results = json.load(file)
                results.append({"train:": str(train), "test": str(test)})
        else:
            results = [{"train:": str(train), "test": str(test)}]
        with open(filename, "w") as outfile:
            json.dump(results, outfile, indent=4)
