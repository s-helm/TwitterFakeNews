import json
import math
import matplotlib.pyplot as plt
from pathlib import Path
from FeatureEngineering.FeatureSelector import get_feature_selection
from FeatureSelection.SelectionUtils import read_var_thresh_results, read_feature_mi_results, \
    read_feature_importance_results
from Learning.EvaluationMetrics import calc_precision, calc_recall
from Learning.LearningUtils import list_to_conf_matr, get_dataset_features
from NLP.NLPUtils import NLPUtils
from Utility.Util import get_root_directory


def save_result(config, filename):
    """
    saves a result to a file. Appends if file already exists
    :param config: 
    :param filename: 
    :return: 
    """

    my_file = Path('results/' + filename + '_params_eval.json')
    if my_file.is_file():
        with open('results/' + filename + '_params_eval.json') as file:
            results = json.load(file)
            results.append(config)
    else:
        results = [config]
    with open('results/' + filename + '_params_eval.json', "w") as outfile:
        json.dump(results, outfile, indent=4)


def read_results(filename, all):
    with open('results/' + filename + '_params_eval.json') as file:
        results = json.load(file)

        results = [res for res in results if res['all'] == all and not math.isnan(res['f1'])]

        # sums = [sum(res['conf_matr']) for res in results]

        res = sorted(results, key=lambda k: k['f1'], reverse=True)
        print("Best parameter tuning result:")
        print(res[0])
        print("F1: {}".format(res[0]['f1']))
        print("Precision: {}".format(calc_precision(list_to_conf_matr(res[0]['conf_matr']))))
        print("Recall: {}".format(calc_recall(list_to_conf_matr(res[0]['conf_matr']))))
        print(list_to_conf_matr(res[0]['conf_matr']))
        return res


def read_results_rf(filename, all, balanced):
    new_res = list()

    res = read_results(filename, all)

    for r in res:
        if balanced:
            if r['params']['class_weight'] == 'balanced':
                new_res.append(r)
        else:
            if r['params']['class_weight'] == None:
                new_res.append(r)
    return new_res


def plot_results(results, title):
    x = []
    y = []
    for res in results:
        x.append(res['params']['n_estimators'])
        y.append(res['f1'])

    y_max_index = y.index(max(y))
    y_max = y[y_max_index]
    x_max = x[y_max_index]
    plt.title(title)
    plt.scatter(x,y)
    plt.scatter(x_max, y_max, color="red")
    plt.xlabel('n_estimators')
    plt.ylabel('F1-score')

    plt.show()


def plot_two_in_one(results_1, title_1, results_2, title_2):
    plt.figure(1)
    plt.subplot(121)
    x_1 = []
    y_1 = []
    for res in results_1:
        x_1.append(res['params']['n_estimators'])
        y_1.append(res['f1'])

    y_1_max_index = y_1.index(max(y_1))
    y_1_max = y_1[y_1_max_index]
    x_1_max = x_1[y_1_max_index]
    plt.title(title_1)
    plt.scatter(x_1,y_1)
    plt.scatter(x_1_max, y_1_max, color="red")
    plt.xlabel('n_estimators')
    plt.ylabel('F1-score')
    # plt.plot(x_1, y_1)
    plt.subplot(122)
    x_2 = []
    y_2 = []
    for res in results_2:
        x_2.append(res['params']['n_estimators'])
        y_2.append(res['f1'])

    y_2_max_index = y_2.index(max(y_2))
    y_2_max = y_2[y_2_max_index]
    x_2_max = x_2[y_2_max_index]
    plt.title(title_2)
    plt.scatter(x_2,y_2)
    plt.scatter(x_2_max, y_2_max, color="red")
    plt.xlabel('n_estimators')
    plt.ylabel('F1-score')
    plt.show()


def print_results_svm(all):
    res = read_results('svm_20k', all=all)

    for r in res:
        print("-------------------------------------------")
        if 'C' in r['params']:
            print("C: {}".format(r['params']['C']))
        if 'svm__C' in r['params']:
            print("C: {}".format(r['params']['svm__C']))
        if 'feature_map__gamma' in r['params']:
            print("gamma: {}".format(r['params']['feature_map__gamma']))
        if 'svm__class_weight' in r['params']:
            print("class weight: {}".format(r['params']['svm__class_weight']))
        if 'class_weight' in r['params']:
            print("class weight: {}".format(r['params']['class_weight']))
        print('kernel: {}'.format(r['kernel']))
        print('f1: {}'.format(r['f1']))


def store_best_config(clf_name, all):
    """
    stores the best configuration of features and parameters in ../Learning/configs/
    :param clf_name: name of the classifier
    :param all: 0: all features, 1: tweet features
    :return: 
    """
    params = None
    features = None
    if all == 1:
        if clf_name == 'nb':
            var_thresh_res = read_var_thresh_results(clf_name, all, print_best=True)[0]
            features = NLPUtils.str_list_to_list(var_thresh_res['features'])
        if clf_name == 'dt':
            features = get_feature_selection(get_dataset_features(clf_name), all=all)
        if clf_name == 'nn':
            mi_res = read_feature_mi_results(clf_name, all=all, print_best=True)[0]
            features = mi_res['features']
            params = read_results('nn_full_set', all=all)[0]
        if clf_name == 'svm':
            mi_res = read_feature_mi_results(clf_name, all=all, print_best=True)[0]
            features = mi_res['features']
            params = read_results('svm_20k_final', all=all)[0]
        if clf_name == 'xgb':
            feat_imp_res = read_feature_importance_results(clf_name, all, print_best=True)[0]
            features = NLPUtils.str_list_to_list(feat_imp_res['features'])
            params = read_results('xgb', all=all)[0]
        if clf_name == 'rf':
            feat_imp_res = read_feature_importance_results(clf_name, all, print_best=True)[0]
            features = NLPUtils.str_list_to_list(feat_imp_res['features'])
            params = read_results('rf', all=all)[0]
    elif all == 0:
        if clf_name == 'nb':
            var_thresh_res = read_var_thresh_results(clf_name, all, print_best=True)[0]
            features = NLPUtils.str_list_to_list(var_thresh_res['features'])
        if clf_name == 'dt':
            feat_imp_res = read_feature_importance_results(clf_name, all, print_best=True)[0]
            features = NLPUtils.str_list_to_list(feat_imp_res['features'])
        if clf_name == 'nn':
            var_thresh_res = read_var_thresh_results(clf_name, all, print_best=True)[0]
            features = NLPUtils.str_list_to_list(var_thresh_res['features'])
            params = read_results('nn_final', all=all)[0]
        if clf_name == 'svm':
            mi_res = read_feature_mi_results(clf_name, all, print_best=True)[0]
            features = mi_res['features']
            params = read_results('svm_100k_final', all=all)[0]
        if clf_name == 'xgb':
            feat_imp_res = read_feature_importance_results(clf_name, all, print_best=True)[0]
            features = NLPUtils.str_list_to_list(feat_imp_res['features'])
        if clf_name == 'rf':
            feat_imp_res = read_feature_importance_results(clf_name, all, print_best=True)[0]
            features = NLPUtils.str_list_to_list(feat_imp_res['features'])

    config = dict()
    if params is not None:
        config['params'] = params['params']

    if clf_name == 'svm':
        config['kernel'] = params['kernel']

    config['features'] = features
    config['clf'] = clf_name
    config['all'] = all


    path = get_root_directory()+'/Learning/configs/' + clf_name + '_best_config.json'
    my_file = Path(path)
    if my_file.is_file():
        with open(path) as file:
            results = json.load(file)
            results.append(config)
    else:
        results = [config]
    with open(path, "w") as outfile:
        json.dump(results, outfile, indent=4)



