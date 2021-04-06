import gc
import json

from matplotlib.ticker import MaxNLocator

from Utility.TimeUtils import TimeUtils
from FeatureEngineering.FeatureSelector import get_feature_selection
from FeatureSelection.SelectionUtils import save_result, read_results
from Learning.LearningMain import build_model, perform_x_val
from Learning.LearningUtils import get_dataset, conf_matr_to_list, get_base_learners, list_to_conf_matr, \
    get_learner_and_features
import matplotlib.pyplot as plt
import numpy as np



def plot_features_importances(clf, data, features):
    """
    plots the relative importances of the features for a given feature set
    :param clf: 
    :param data: 
    :param features: 
    :return: 
    """
    feat_imps = get_feature_importances(clf, data, features)
    plot_feature_importance(feat_imps)


def get_feature_importances(clf, data, features):
    """
    learns a model on the data to get the feature importances
    :param clf: classifier to use
    :param data: 
    :param features: 
    :return: 
    """
    print("Learn model to get feature importances...")
    build_model(data, features=features, clf=clf)
    feat_imps = clf.feature_importances_
    return get_feature_score_list(features, feat_imps)


def remove_n_worst(feat_imps, n):
    """
    removes the n items with the worst score
    :param feat_imps: 
    :param n: 
    :return: 
    """
    copy = feat_imps[:]
    copy = sorted(copy, key=lambda tup: tup[1])
    copy = copy[n:]

    return [feat for feat in feat_imps if feat[0] in [i[0] for i in copy]]


def select_features_on_importances(clf_name, data, all, n=10):
    """
    recursively removes the n feature with the least importance
    :param clf: classifier to use
    :param data: 
    :param all: 
    :param n: 
    :param var_thresh: 
    :return: 
    """

    if all == 0:
        filename = clf_name + '_with_user_selection'
    else:
        filename = clf_name+'_wo_user_selection'

    clf = get_base_learners(clf_name)
    features = get_feature_selection(data, all=all)
    # if var_thresh:
    #     to_remove = get_variance_selection(clf_name, all=all)
    #     features = [f for f in features if f not in to_remove]

    feat_imps = get_feature_importances(clf, data, features)

    # remove all with 0 importance
    feat_imps = [i for i in feat_imps if i[1] > 0]

    print(feat_imps)

    features = [i[0] for i in feat_imps]
    f1 = evaluate(clf, data, features, all=all, filename=filename)

    old_f1 = f1
    while True:

        # remove all with 0 importance
        feat_imps = [i for i in feat_imps if i[1] > 0]
        # remove n least important features
        features = [i[0] for i in remove_n_worst(feat_imps, n)]
        if len(features) == 0:
            n = 1
            features = [i[0] for i in remove_n_worst(feat_imps, n)]

        # evaluate
        f1 = evaluate(clf, data, features, all=all, filename=filename)

        if f1 >= old_f1:
            old_f1 = f1
            feat_imps = get_feature_importances(clf, data, features)
        else:
            if n > 1:
                print('F1 decreased -> set n=1')
                n = 1
            else:
                print('F1 decreased and n=1 -> stop.')
                break
    gc.collect()

def evaluate(clf, data, features, all, filename, standardize=True):
    """
    evaluates a model using cross validation and stores the results
    :param clf: classifer to use
    :param data: 
    :param features: 
    :param filename: 
    :return: 
    """

    timebefore = TimeUtils.get_time()
    f1, conf_matr = perform_x_val(data, clf, features=features, standardize=standardize)
    timeafter = TimeUtils.get_time()
    time_diff = timeafter - timebefore

    conf = dict()
    conf['f1'] = f1
    conf['conf_matr'] = conf_matr_to_list(conf_matr)
    conf['time_diff'] = str(time_diff)
    conf['features'] = str(features)
    conf['num_features'] = len(features)
    conf['all'] = all

    save_result(conf, filename=filename)
    return f1

def plot_feature_importance(map, top_n=20, show=True):
    """
    plots the relative feature importance
    :param map: map with feature importances
    :param top_n: top n features
    :return: 
    """
    imp_sum = sum([i[1] for i in map])
    # relative feature importance
    map = [(i[0],i[1]/imp_sum) for i in map]

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

    ax1.set_title("Relative Feature Importance")

    ax1.set_xlim([0, scores[len(scores)-1]+0.2])
    ax1.xaxis.set_major_locator(MaxNLocator(11))
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)

    # set X-axis tick marks at the deciles
    imp = ax1.text(.5, -.07, 'Relative Importance',
                            horizontalalignment='center', size='small',
                            transform=ax1.transAxes)


    for rect in rects:
        ax1.text(rect.get_width() + 0.01, rect.get_y() + rect.get_height()/2.,
                '{}'.format(rect.get_width()),
                ha='left', va='center')

    if show:
        plt.show()
    else:
        return plt


def get_feature_score_list(features, scores):
    """
    creates a list of tuples from the features and their importances
    :param features: features
    :param scores: features importances
    :return: list of (feature,score)
    """
    map = list()
    if len(features) == len(scores):
        for i in range(len(features)):
            map.append((features[i],scores[i]))
    else:
        raise IndexError('Length of features and scores are not equal. {} != {}'.format(len(features), len(scores)))
    # return sorted(map, key=lambda tup: tup[1])
    return map

def store_feature_importance(clf_name, all):
    file = 'importances/features_importance_{}_{}.json'.format(clf_name, all)
    clf, features = get_learner_and_features(clf_name=clf_name, all=all)
    feat_imps = get_feature_importances(clf, get_dataset(clf_name), features)
    feat_imps = sorted(feat_imps, key=lambda tup: tup[1], reverse=True)
    feat_imps = [(i[0],i[1].item()) for i in feat_imps]

    with open(file, 'w') as file:
        json.dump(feat_imps, file)

def read_features_importance_from_file(clf_name, all, print_imps=True):
    """
    read feature importances from file
    :param clf_name: 
    :param all: 
    :param print_imps: if true, prints the importance map
    :return: 
    """
    file = 'importances/features_importance_{}_{}.json'.format(clf_name, all)
    with open(file) as json_data:
        map = json.load(json_data)

    map = sorted(map, key=lambda tup: tup[1], reverse=True)
    if print_imps:
        for i, m in enumerate(map, 1):
            print("{}. {}: {}".format(i, m[0],m[1]))
        return map
    else:
        return map


def plot_feature_importance_from_file(clf_name, all, save=False):
    """
    reads importances from file and plots them
    :param clf_name: 
    :param all: 
    :param save: if true, saves the files
    :return: 
    """
    map = read_features_importance_from_file(clf_name, all, print_imps=False)

    plt = plot_feature_importance(map, top_n=20, show=False)

    if save:
        plt.savefig('importances/fi_after_tuning_{}_{}.pdf'.format(clf_name, all))
    else:
        plt.show()

if __name__ == "__main__":
    # perform selection
    # clfs = ['dt','rf','xgb']
    #
    # # tweet features only
    # for clf_name in clfs:
    #     data = get_dataset(clf_name)
    #     select_features_on_importances(clf_name, data, all=1)
    #     gc.collect()
    #
    # # tweet+user features
    # for clf_name in clfs:
    #     data = get_dataset(clf_name)
    #     select_features_on_importances(clf_name, data, all=0)
    #     gc.collect()

    # read feature importances
    map = read_features_importance_from_file('xgb', all=1)

    # creates feature importances plots
    # all = 1
    # clfs = ['dt','xgb','rf']
    # for clf_name in clfs:
    #     plot_feature_importance_from_file(clf_name, all, save=False)




