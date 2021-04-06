import json
import os
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from FeatureEngineering.FeatureSelector import get_feature_selection
from Learning.EvaluationMetrics import calc_precision, calc_recall
from Learning.LearningUtils import get_base_learners, list_to_conf_matr, get_clf_names, get_dataset_features
from NLP.NLPUtils import NLPUtils
from Utility.Util import get_root_directory


def save_result(config, filename):
    """
    saves a result to a file. Appends if file already exists
    :param config: 
    :param filename: 
    :return: 
    """
    directory = "results/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    my_file = Path(directory + filename + '.json')
    if my_file.is_file():
        with open(directory + filename + '.json') as file:
            results = json.load(file)
            results.append(config)
    else:
        results = [config]
    with open(directory + filename + '.json', "w") as outfile:
        json.dump(results, outfile, indent=4)


def read_results(filename, print_best=False):
    with open(get_root_directory() +'/FeatureSelection/results/' + filename + '.json') as file:
        results = json.load(file)

        # results = [res for res in results if res['all'] == all]

        res = sorted(results, key=lambda k: k['f1'], reverse=True)
        if print_best:
            print("Best result with use case {}: ".format(2 - res[0]['all']))
            print(res[0])
        return res


def read_feature_importance_results(clf_name, all, print_best=False):
    """
    reads the results from the feature selection by feature importance
    :param clf: clf name
    :param all: 1: tweet features only, 0: all features
    :param print_best: if True, print best result
    :return: 
    """
    if all == 1:
        filename = clf_name+'_wo_user_selection'
    else:
        filename = clf_name+'_with_user_selection'
    res = read_results(filename)

    if print_best:
        print("Best result with use case {}: ".format(2-res[0]['all']))
        print("F1: {}".format(res[0]['f1']))
        print("Precision: {}".format(calc_precision(list_to_conf_matr(res[0]['conf_matr']))))
        print("Recall: {}".format(calc_recall(list_to_conf_matr(res[0]['conf_matr']))))
        print(list_to_conf_matr(res[0]['conf_matr']))
        print("Features: {}".format(len(NLPUtils.str_list_to_list(res[0]['features']))))
    return res


def read_feature_mi_results(clf_name, all, print_best=False, sample=False):
    """
    reads the results from the feature selection by mutual information
    :param clf: clf name
    :param all: 1: tweet features only, 0: all features
    :param print_best: if True, print best result
    :return: 
    """
    if all == 1:
        if sample:
            filename = clf_name+'_mi_wo_user'
        else:
            filename = clf_name+'_mi_wo_user_on_full_set'
    else:
        if sample:
            filename = clf_name+'_mi_with_user'
        else:
            filename = clf_name+'_mi_with_user_on_full_set'
    res = read_results(filename)

    if print_best:
        print("Best result MI with use case {}: ".format(2-res[0]['all']))
        print("F1: {}".format(res[0]['f1']))
        print("Precision: {}".format(calc_precision(list_to_conf_matr(res[0]['conf_matr']))))
        print("Recall: {}".format(calc_recall(list_to_conf_matr(res[0]['conf_matr']))))
        print(list_to_conf_matr(res[0]['conf_matr']))
        print("Features: {}".format(len(res[0]['features'])))
    return res

def read_var_thresh_results(clf, all, print_best=False):
    """
    reads the variance threshold results
    :param clf: clf name
    :param all: 1: tweet features only, 0: all features
    :param print_best: if True, print best result
    :return: 
    """

    if all == 1:
        filename = 'variance_thresholds_wo_user'
    else:
        filename = 'variance_thresholds_with_user'

    res = read_results(filename)

    clf_res = list()
    for re in res:
        if re['clf'] == type(get_base_learners(clf)).__name__:
            clf_res.append(re)

    res = sorted(clf_res, key=lambda k: k['f1'], reverse=True)

    if print_best:
        print("Best result with use case {}: ".format(2-res[0]['all']))
        print("Threshold: {}".format(res[0]['threshold']))
        print("F1: {}".format(res[0]['f1']))
        print("Precision: {}".format(calc_precision(list_to_conf_matr(res[0]['conf_matr']))))
        print("Recall: {}".format(calc_recall(list_to_conf_matr(res[0]['conf_matr']))))
        print(list_to_conf_matr(res[0]['conf_matr']))
        print("Features: {}".format(len(NLPUtils.str_list_to_list(res[0]['features']))))
    return res


def read_feature_mis(clf, all, top_n):
    """
    reads the file that contains the mutual information scores for each feature
    :param clf: name of clf
    :param all: 1: use case 1, 0: use case 2
    :param top_n: number of top features to return
    :return: 
    """
    print('Load mutual information from file')
    file = 'results/mutual_information_{}.json'.format(clf)
    with open(file) as json_data:
        d = json.load(json_data)
        mi = [(feat[0], feat[1]) for feat in d]

        features = get_feature_selection(get_dataset_features(clf), all=all)
        mi = [i for i in mi if i[0] in features]
        mi.sort(key=lambda tup: tup[1], reverse=True)

        print("Top {} features: ".format(top_n))
        for n in range(top_n):
            print("{}. {}: {}".format((n+1),mi[n][0],mi[n][1]))
        return mi[:top_n]


def get_baseline_result(clf_name, all, print_result=False):
    """
    returns the baseline f1 measure (=variance selection with all features removed that have 0 variance)
    :param clf_name: 
    :param all: 
    :return: 
    """
    res = read_var_thresh_results(clf_name, all)
    for re in res:
        if re['threshold'] == 0:
            if print_result:
                print("Baseline with use case {}: ".format(2 - re['all']))
                print("Threshold: {}".format(re['threshold']))
                print("F1: {}".format(re['f1']))
                print("Precision: {}".format(calc_precision(list_to_conf_matr(re['conf_matr']))))
                print("Recall: {}".format(calc_recall(list_to_conf_matr(re['conf_matr']))))
                print(list_to_conf_matr(re['conf_matr']))
                print("Features: {}".format(len(NLPUtils.str_list_to_list(re['features']))))

            return re['f1']


def get_best_result(res, print_best=True):
    res = sorted(res, key=lambda k: k['f1'], reverse=True)

    if print_best:
        print("Best: ")
        if 'threshold' in res[0]:
            print("Threshold: {}".format(res[0]['threshold']))
        print("F1: {}".format(res[0]['f1']))
        print("Precision: {}".format(calc_precision(list_to_conf_matr(res[0]['conf_matr']))))
        print("Recall: {}".format(calc_recall(list_to_conf_matr(res[0]['conf_matr']))))
        print(list_to_conf_matr(res[0]['conf_matr']))
        print("Features: {}".format(len(NLPUtils.str_list_to_list(res[0]['features']))))
    return res[0]


def print_result(res):
    if 'threshold' in res:
        print("Threshold: {}".format(res['threshold']))
    print("F1: {}".format(res['f1']))
    print("Precision: {}".format(calc_precision(list_to_conf_matr(res['conf_matr']))))
    print("Recall: {}".format(calc_recall(list_to_conf_matr(res['conf_matr']))))
    print(list_to_conf_matr(res['conf_matr']))
    print("Features: {}".format(len(NLPUtils.str_list_to_list(res['features']))))


def get_best_feature_set(clf_name, all):
    """
    reads the results of the variance, feature importance and mutual information feature selections and returns the best value
    :param clf_name: name of clf
    :param all: 1: tweet features, 0: tweet+user features
    :return: features
    """
    best_res = dict()

    var_thresh_res = read_var_thresh_results(clf_name, all)[0]
    best_res['var'] = var_thresh_res['f1']

    mi_res = read_feature_mi_results(clf_name, all, sample=False)[0]
    best_res['mi'] = mi_res['f1']

    if clf_name in ['dt','rf','xgb']:
        feat_imp_res = read_feature_importance_results(clf_name, all)[0]
        best_res['imp'] = feat_imp_res['f1']

    best = max(best_res, key=best_res.get)
    if best == 'imp':
        print("Use feature importance selection (F1: {})".format(feat_imp_res['f1']))
        print_result(feat_imp_res)
        return NLPUtils.str_list_to_list(feat_imp_res['features'])
    if best == 'var':
        print("Use variance selection (F1: {})".format(var_thresh_res['f1']))
        print_result(var_thresh_res)
        return NLPUtils.str_list_to_list(var_thresh_res['features'])
    if best == 'mi':
        print("Use mutual info selection (F1: {})".format(mi_res['f1']))
        mi_res['features'] = str(mi_res['features'])
        print_result(mi_res)
        return NLPUtils.str_list_to_list(mi_res['features'])


def print_var_thresh_latex_table(res, best):

    if best:
        res = sorted(res, key=lambda k: k['f1'], reverse=True)[0]

        prec = "{0:.4f}".format(float(calc_precision(list_to_conf_matr(res['conf_matr']))))
        rec = "{0:.4f}".format(float(calc_recall(list_to_conf_matr(res['conf_matr']))))
        f1 = "{0:.4f}".format(float(res['f1']))

        print("{} & {} & {} & {} & {} & {} \\\\\\hline".format(res['clf'], res['threshold'], len(NLPUtils.str_list_to_list(res['features'])),
                                                   prec,
                                                   rec,
                                                   f1
                                                   ))
    else:
        for re in res:
            prec = "{0:.4f}".format(float(calc_precision(list_to_conf_matr(re['conf_matr']))))
            rec = "{0:.4f}".format(float(calc_recall(list_to_conf_matr(re['conf_matr']))))
            f1 = "{0:.4f}".format(float(re['f1']))
            print("{} & {} & {} & {} & {} & {} \\\\\\hline".format(re['clf'], re['threshold'],
                                                       len(NLPUtils.str_list_to_list(re['features'])),
                                                       prec,
                                                       rec,
                                                       f1
                                                       ))


def get_feature_usages(all):
    clf_names = get_clf_names()

    feat_coll = list()
    for clf in clf_names:
        feat_coll.extend(get_best_feature_set(clf, all))
    counter = Counter(feat_coll)
    print(counter.most_common(50))
    return counter


def get_features_removed(res, clf_name):
    """
    returns the features that were removed in 
    :param res: result read from json file
    :param clf_name: 
    :return: 
    """
    data = get_dataset_features(clf_name)
    orig_feats = get_feature_selection(data=data, all=res['all'])

    feats_sel = NLPUtils.str_list_to_list(res['features'])

    removed = [feat for feat in orig_feats if feat not in feats_sel]
    print("Features removed: {}".format(removed))
    return removed


def plot_feature_mi(map, top_n=20):
    """
    plots the relative feature importance
    :param map: map with feature importances
    :param top_n: top n features
    :return: 
    """
    map = sorted(map, key=lambda tup: tup[1])
    map = map[len(map)-top_n:]
    features = [x[0] for x in map]
    scores = [x[1] for x in map]

    fig, ax1 = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.115, right=0.88)
    fig.canvas.set_window_title('Eldorado K-8 Fitness Chart')
    pos = np.arange(len(features))

    rects = ax1.barh(pos, [scores[k] for k in range(len(scores))],
                     align='center',
                     height=0.2, color='b',
                     tick_label=features)

    ax1.set_title("Mutual Information")

    ax1.set_xlim([0, scores[len(scores)-1]+0.2])
    ax1.xaxis.set_major_locator(MaxNLocator(11))
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)

    # set X-axis tick marks at the deciles
    imp = ax1.text(.5, -.07, 'Mutual Information',
                            horizontalalignment='center', size='small',
                            transform=ax1.transAxes)


    for rect in rects:
        ax1.text(rect.get_width() + 0.01, rect.get_y() + rect.get_height()/2.,
                '{}'.format(rect.get_width()),
                ha='left', va='center')

    plt.show()

if __name__ == "__main__":
    # read feature selection results
    clf_name = 'svm'
    all = 1

    # read results
    # res_var = read_var_thresh_results(clf_name, all)
    res_feat = read_feature_importance_results(clf_name, all)
    # res_mi = read_feature_mi_results(clf_name, all, print_best=True)
    # get_baseline_result(clf_name, all, print_result=True)

    # plot mutual information
    mi_map = read_feature_mis(clf_name, all=all, top_n=400)
    plot_feature_mi(mi_map)
