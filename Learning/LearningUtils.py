import csv
import json
from pathlib import Path

import numpy as np
from sklearn import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from Utility.CSVUtils import load_data_from_CSV, read_header_from_CSV
from Utility.Util import get_root_directory


def load_model(filename):
    return joblib.load('models/' + filename + '.pkl')


def get_clf_names():
    return ['nb', 'dt', 'svm', 'nn', 'rf', 'xgb']


def get_sklearn_clf_names(name=None):
    if name is None:
        names = get_clf_names()
        return [type(get_base_learners(name)).__name__ for name in names]
    else:
        return type(get_base_learners(name)).__name__


def get_raw_dataset():
    datasets = get_root_directory()+"/data/"
    return load_data_from_CSV(datasets + 'data_set_tweet_user_features.csv')


def get_dataset(clf):
    datasets = get_root_directory()+"/data/"

    return load_data_from_CSV(datasets + 'data_set_' + clf + '.csv')


def get_testset(clf, testset_only=False):
    """
    returns the test dataset
    :param clf: if True, the test dataset with topic and text representation trained on the testset is used
    :return:
    """
    datasets = get_root_directory() + "/data/testdata/"
    if testset_only:
        return load_data_from_CSV(datasets + 'testset_only_' + clf + '.csv')
    else:
        return load_data_from_CSV(datasets + 'testset_' + clf + '.csv')


def get_dataset_features(clf):
    """
    returns the header of a dataset to retrieve the feature names
    :param clf: 
    :return: 
    """
    datasets = get_root_directory() + "/data/"
    return read_header_from_CSV(datasets + 'data_set_' + clf + '.csv')


def save_result(config, filename):
    """
    saves a result to a file. Appends if file already exists
    :param config: 
    :param filename: 
    :return: 
    """

    my_file = Path(filename + '.json')
    if my_file.is_file():
        with open(filename + '.json') as file:
            results = json.load(file)
            results.append(config)
    else:
        results = [config]
    with open(filename + '.json', "w") as outfile:
        json.dump(results, outfile, indent=4)


def get_learner_names():
    return ['nb', 'dt', 'svm', 'nn', 'rf', 'xgb']


def get_base_learners(name=None):
    clf_nb = GaussianNB()
    clf_dt_gini = DecisionTreeClassifier()
    clf_neural_network = MLPClassifier()
    clf_svm_lin = LinearSVC()
    clf_xgb = XGBClassifier()
    clf_rf = RandomForestClassifier()

    if name is None:
        return [clf_nb, clf_dt_gini, clf_neural_network, clf_xgb, clf_svm_lin, clf_rf]
    else:
        if name == 'nb':
            return clf_nb
        if name == 'dt':
            return clf_dt_gini
        if name == 'nn':
            return clf_neural_network
        if name == 'svm':
            return clf_svm_lin
        if name == 'xgb':
            return clf_xgb
        if name == 'rf':
            return clf_rf
        else:
            return None


def get_kernel_approx(features, gamma=0, approx='nystroem', params=None):
    """
    returns a pipeline for kernel approximation
    :param features: 
    :param gamma: 
    :param approx: approximation method. either 'rbf' for RBFSampler or 'nystroem' for Nystroem approximation
    :param params: additional svm parameters
    :return: 
    """
    clf = LinearSVC()
    if params is not None:
        clf.set_params(**params)
    if approx == 'rbf':
        feature_map_rbf = RBFSampler()
        rbf_approx_svm = pipeline.Pipeline([("feature_map", feature_map_rbf), ("svm", clf)])
        rbf_approx_svm.set_params(feature_map__n_components=len(features), feature_map__gamma=gamma)
        return rbf_approx_svm
    elif approx == 'nystroem':
        feature_map_nystroem = Nystroem()
        nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem), ("svm", clf)])
        nystroem_approx_svm.set_params(feature_map__n_components=len(features), feature_map__gamma=gamma)
        return nystroem_approx_svm


def append_to_results_csv(row, filename):
    """
    appends at the end of the results csv file
    :param row: 
    :param filename: 
    :return: 
    """
    f = open(filename, 'a')
    try:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(row)
    finally:
        f.close()


def conf_matr_to_list(conf_matr):
    return [float(conf_matr[0, 0]), float(conf_matr[0, 1]), float(conf_matr[1, 0]), float(conf_matr[1, 1])]


def list_to_conf_matr(conf_matr_list):
    conf_matr = np.zeros((2, 2))
    conf_matr[0, 0] = conf_matr_list[0]
    conf_matr[0, 1] = conf_matr_list[1]
    conf_matr[1, 0] = conf_matr_list[2]
    conf_matr[1, 1] = conf_matr_list[3]
    return conf_matr


def get_learner_and_features(clf_name, all):
    """
    returns the learner with the best parameter configuration and the best feature set
    The configurations are loaded from ../configs/
    :param clf_name: name of classifier
    :param all: use case (0: uc 2, 1: uc 1)
    :return: 
    """

    with open(get_root_directory() + '/Learning/configs/' + clf_name + '_best_config.json') as file:
        results = json.load(file)

    res = [r for r in results if r['all'] == all and r['clf'] == clf_name][0]

    print("------------------------------")
    if 'params' in res:
        print("Classifier: " + clf_name)
        print("Number of features: " + str(len(res['features'])))
        print("Parameters: {}".format(res['params']))

        if clf_name == 'svm':
            if res['kernel'] == 'linear':
                return LinearSVC(C=res['params']['C']), res['features']
            else:
                print("Kernel: " + res['kernel'])
                return get_kernel_approx(res['features'], approx=res['kernel'],
                                         gamma=res['params']['feature_map__gamma'],
                                         params={'C': res['params']['svm__C']}), res['features']
        elif clf_name == 'xgb':
            xgb = XGBClassifier()
            return xgb.set_params(**res['params']), res['features']
        elif clf_name == 'rf':
            rf = RandomForestClassifier()
            return rf.set_params(**res['params']), res['features']
        elif clf_name == 'nn':
            nn = MLPClassifier()
            return nn.set_params(**res['params']), res['features']
        else:
            return get_base_learners(clf_name), res['features']
    else:
        print('No parameters given. Return default learner.')
        print("Classifier: " + clf_name)
        print("Number of features: " + str(len(res['features'])))
        return get_base_learners(clf_name), res['features']
