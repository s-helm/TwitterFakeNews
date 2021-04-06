import copy
import random
from threading import Thread
import numpy as np
import pandas as pd
import gc
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from DatasetUtils.Standardizer import standardize_data
from FeatureEngineering.FeatureSelector import get_group_feature, get_label, get_feature_selection
from Learning.EvaluationMetrics import print_result, get_result, calc_f1
from Learning.LearningUtils import get_dataset, get_base_learners, save_result, get_learner_and_features
from Utility.CSVUtils import save_df_as_csv, load_data_from_CSV


class Voting:
    def __init__(self, clf_list, all, weighted=False, n_splits=10):
        self.clf_list = clf_list
        self.all = all
        self.weighted = weighted
        self.n_splits = n_splits
        self.ys = list()
        self.tweet_ids = list()

        self.preds = [None]*n_splits
        self.conf_matrs = [list()]*n_splits
        self.conf_matr = np.zeros((2,2))
        self.conf_matr_stack = np.zeros((2,2))

    def fit(self, store=True):
        """
        performs a X-Validation to get the results from models
        :param store: if True, predictions get stored in a dataframe with a name that is set by 'store'
        :return: 
        """

        data = get_dataset(self.clf_list[0])
        print(data.shape)
        X = data[get_feature_selection(data, all=all)].as_matrix()
        ids = data['tweet__id'].as_matrix()
        y = data[get_label()].as_matrix()
        groups = data[get_group_feature()].as_matrix()

        gkf = GroupKFold(self.n_splits)

        # print(groups.shape)
        # create the training and testing indices
        indices_train = []
        indices_test = []
        for train, test in gkf.split(X, y, groups=groups):
            # X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            indices_train.append(train)
            indices_test.append(test)
            self.ys.append(y[test])
            self.tweet_ids.append(ids[test])
        del data
        del X
        del y
        gc.collect()

        for clf_conf in self.clf_list:
            data = get_dataset(clf_conf)
            clf, features = get_learner_and_features(clf_name=clf_conf, all=self.all)

            X = data[features].as_matrix()
            y = data[get_label()].as_matrix()

            del data

            i = 0
            threads_coll = []
            n_splits = len(list(zip(indices_train, indices_test)))
            for train, test in list(zip(indices_train, indices_test)):
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

                if (isinstance(clf, GaussianNB) or
                                        isinstance(clf, MLPClassifier) or
                                        isinstance(clf, LinearSVC) or
                                        isinstance(clf, Pipeline)):
                    print("Standardize Data")
                    X_train, X_test = standardize_data(X_train, X_test)

                clf_i = copy.deepcopy(clf)
                t = Thread(target=self.train_and_predict, args=(clf_i, X_train, X_test, y_train, y_test, i))
                threads_coll.append(t)
                i += 1
                nr_of_cores = 2

                if i % nr_of_cores == 0 or i == n_splits:
                    # Start all threads
                    for x in threads_coll:
                        x.start()

                    # Wait for all of them to finish
                    for x in threads_coll:
                        x.join()
                        print("Thread {} done.".format(x))
                    threads_coll = list()
            del X
            del y
            gc.collect()

        # append labels and tweet ids
        for i in range(self.n_splits):
            cols = [col for col in list(self.preds[i].columns)]
            # append label
            if get_label() not in cols:
                self.preds[i][get_label()] = list(self.ys[i])
            # append tweet ids
            if 'tweet__id' not in cols:
                self.preds[i]['tweet__id'] = list(self.tweet_ids[i])
            if store:
                # save the resulting df of each fold
                if self.weighted:
                    save_df_as_csv(self.preds[i], 'voting_folds/preds_fold_weighted_{}_{}.csv'.format(self.all, i))
                else:
                    save_df_as_csv(self.preds[i], 'voting_folds/preds_fold_{}_{}.csv'.format(self.all, i))

    def train_and_predict(self, model, X_train, X_test, y_train, y_test, i=None):
        """trains a single model and appends its prediction to the global prediction"""

        model.fit(X_train, y_train)

        if self.weighted:
            if isinstance(model, LinearSVC) or isinstance(model, Pipeline):
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict_proba(X_test)
        else:
            y_pred = model.predict(X_test)
            if type(model) == XGBClassifier:
                y_pred = [round(value) for value in y_pred]

        if not self.weighted:
            if self.preds[i] is None:
                self.preds[i] = pd.DataFrame({type(model).__name__:list(y_pred)})
            else:
                self.preds[i][type(model).__name__] = list(y_pred)

            self.conf_matrs[i].append(np.array(confusion_matrix(y_test, y_pred, [1, 0])))
        else:
            if self.preds[i] is None:
                if len(y_pred.shape) == 2:
                    self.preds[i] = pd.DataFrame({"{}_{}".format(type(model).__name__,0):list(y_pred[:,0])})
                    self.preds[i]["{}_{}".format(type(model).__name__,1)] = list(y_pred[:,1])
                elif len(y_pred.shape) == 1:
                    # probability of class 0/1 is 1 or 0, repsectively
                    y_pred_0 = [int(pred == 0) for pred in y_pred]
                    y_pred_1 = [int(pred == 1) for pred in y_pred]
                    self.preds[i] = pd.DataFrame({"{}_{}".format(type(model).__name__,0): y_pred_0})
                    self.preds[i]["{}_{}".format(type(model).__name__,1)] = list(y_pred_1)
            else:
                if len(y_pred.shape) == 2:
                    self.preds[i]["{}_{}".format(type(model).__name__,0)] = list(y_pred[:,0])
                    self.preds[i]["{}_{}".format(type(model).__name__,1)] = list(y_pred[:,1])
                elif len(y_pred.shape) == 1:
                    # probability of class 0/1 is 1 or 0, repsectively
                    y_pred_0 = [int(pred == 0) for pred in y_pred]
                    y_pred_1 = [int(pred == 1) for pred in y_pred]
                    self.preds[i]["{}_{}".format(type(model).__name__,0)] = list(y_pred_0)
                    self.preds[i]["{}_{}".format(type(model).__name__,1)] = list(y_pred_1)

        del X_train
        del X_test
        del y_train
        del y_test
        gc.collect()

    def vote(self, learners=None):
        """
        performs the voting
        :return: 
        """
        self.conf_matr = np.zeros((2,2))

        for i in range(self.n_splits):
            if 'vote' in self.preds[i].columns:
                self.preds[i].drop('vote', 1, inplace=True)
            if 'vote_res' in self.preds[i].columns:
                self.preds[i].drop('vote_res', 1, inplace=True)

            if learners is None:
                cols = [col for col in list(self.preds[i].columns) if col != 'tweet__id' and col != get_label()]
            else:
                cols = [col for col in list(self.preds[i].columns) if col in learners]

            self.preds[i]['vote'] = self.preds[i][cols].mean(axis=1)
            self.preds[i]['vote_res'] = self.preds[i]['vote'].map(lambda x: self.determine_class(x))
            pred = self.preds[i]['vote_res'].tolist()

            self.conf_matr = np.add(self.conf_matr, np.array(confusion_matrix(self.ys[i], pred, [1, 0])))

        if learners is None:
            print("Voting result with {}: ".format(self.clf_list))
            print_result(self.conf_matr)
            res = get_result(self.conf_matr)
            res['all'] = self.all
            res['weighted'] = self.weighted
            save_result(res, "results/voting_results")
        else:
            return self.conf_matr

    def weighted_vote(self, learners=None):
        """
        performs the weighted voting
        :return: 
        """
        self.conf_matr = np.zeros((2,2))

        for i in range(self.n_splits):
            if 'vote' in self.preds[i].columns:
                self.preds[i].drop('vote', 1, inplace=True)
            if 'vote_res' in self.preds[i].columns:
                self.preds[i].drop('vote_res', 1, inplace=True)

            if learners is None:
                cols = [col for col in list(self.preds[i].columns) if col != 'tweet__id' and col != get_label()]
            else:
                cols = [col for col in list(self.preds[i].columns) if col[:-2] in learners]

            cols_0 = [col for col in cols if '0' in col]
            cols_1 = [col for col in cols if '1' in col]

            self.preds[i]['w_avg_0'] = self.preds[i][cols_0].mean(axis=1)
            self.preds[i]['w_avg_1'] = self.preds[i][cols_1].mean(axis=1)
            self.preds[i]['vote_res'] = self.preds[i].apply(lambda x: self.determine_class_weighted(x['w_avg_0'], x['w_avg_1']),axis=1)
            pred = self.preds[i]['vote_res'].tolist()

            self.conf_matr = np.add(self.conf_matr, np.array(confusion_matrix(self.ys[i], pred, [1, 0])))

        if learners is None:
            print("Weighted voting result with {}: ".format(self.clf_list))
            print_result(self.conf_matr)
            res = get_result(self.conf_matr)
            res['all'] = self.all
            res['weighted'] = self.weighted
            save_result(res, "results/voting_results")
        else:
            return self.conf_matr


    def find_best_voting_combi(self):
        """
        votes all combinations of at least two learners.
        Stores best result.
        :return: 
        """
        learners_cols = [col for col in self.preds[0].columns if col != 'tweet__id' and col != 'tweet__fake']

        best_comb = None
        best_f1 = 0
        best_conf_matr = None

        if self.weighted:
            learners_cols = set([col[:-2] for col in learners_cols])

            for i in range(2, len(learners_cols) + 1):
                combs = itertools.combinations(learners_cols, i)

                for comb in combs:
                    # new_comb = list()
                    # comb_0 = [c+'_0' for c in comb]
                    # comb_1 = [c+'_1' for c in comb]
                    # new_comb.extend(comb_0)
                    # new_comb.extend(comb_1)
                    conf_matr = self.weighted_vote(comb)
                    f1 = calc_f1(conf_matr)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_comb = comb
                        best_conf_matr = conf_matr
        else:
            for i in range(2, len(learners_cols) + 1):
                combs = itertools.combinations(learners_cols, i)
                for comb in combs:
                    conf_matr = self.vote(comb)
                    f1 = calc_f1(conf_matr)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_comb = comb
                        best_conf_matr = conf_matr

        print("Best combination: {}".format(best_comb))
        print_result(best_conf_matr)
        best = get_result(best_conf_matr)
        best['comb'] = best_comb
        best['all'] = self.all
        best['weighted'] = self.weighted
        save_result(best, "results/voting_combi_results")


    @staticmethod
    def determine_class(x):
        """
        determines the class of the result
        :param x: 
        :return: 
        """
        if x == 0.5:
            return int(bool(random.getrandbits(1)))
        else:
            return int(x>0.5)

    @staticmethod
    def determine_class_weighted(x_0, x_1):
        """
        
        :param x_0: average weight class 0
        :param x_1: average weight class 1
        :return: classification result
        """
        if x_0 > x_1:
            return 0
        elif x_0 < x_1:
            return 1
        else:
            return int(bool(random.getrandbits(1)))

    def read_preds_from_df(self):
        """
        reads the dataframes with the predictions that were stored through fitting to determine results
        :return: 
        """
        if self.weighted:
            self.ys = [list()] * self.n_splits
            for i in range(self.n_splits):
                self.preds[i] = load_data_from_CSV('voting_folds/preds_fold_weighted_{}_{}.csv'.format(self.all, i))
                self.ys[i] = self.preds[i][get_label()]
        else:
            self.ys = [list()] * self.n_splits
            for i in range(self.n_splits):
                self.preds[i] = load_data_from_CSV('voting_folds/preds_fold_{}_{}.csv'.format(self.all, i))
                self.ys[i] = self.preds[i][get_label()]


if __name__ == "__main__":

    clf_list = ['nb','dt','svm','nn','xgb','rf']
    voting = Voting(clf_list=clf_list, all=1, weighted=False)
    # voting.read_preds_from_df()
    voting.fit()
    # voting.vote()
    voting.find_best_voting_combi()
