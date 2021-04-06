import gc
import json
import os

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from FeatureEngineering.FeatureSelector import get_feature_selection, get_label, get_group_feature
from FeatureSelection.SelectionMethods import SelectionMethods
from FeatureSelection.SelectionUtils import save_result, get_baseline_result, read_feature_mi_results
from Learning.LearningMain import perform_x_val
from Learning.LearningUtils import get_base_learners, conf_matr_to_list, get_dataset
from Utility.CSVUtils import load_data_from_CSV
from Utility.TimeUtils import TimeUtils
from Utility.Util import list_diff, get_root_directory


def perform_feature_selection_variance_threshold(all):
    clfs = ['svm', 'nn', 'nb', 'dt', 'rf', 'xgb']

    for clf in clfs:
        data = get_dataset(clf)
        print(data.columns.values)
        features = get_feature_selection(data, all=all)
        X = data[features]
        y = data[get_label()]
        user_id = data[get_group_feature()]
        data = None
        print("before: " + str(X.shape))

        before = list(X.columns)
        print("tweet__fake in before: " + str("tweet__fake" in before))

        thresh = 0
        f1_prev = 0
        for i in range(5):
            X = SelectionMethods.variance_threshold_selector(X, thresh)

            print("after: " + str(X.shape))
            after = list(X.columns)
            print("tweet__fake in after: " + str("tweet__fake" in after))
            removed_features = list_diff(before, after)
            print("Features removed (" + str(len(before) - len(after)) + "): " + str(removed_features))

            features = list_diff(before, removed_features)
            print("tweet__fake in features: " + str("tweet__fake" in features))

            # add column with label
            df = pd.concat([X, y], axis=1)
            df = pd.concat([df, user_id], axis=1)

            clf_mod = get_base_learners(name=clf)

            timebefore = TimeUtils.get_time()
            f1, conf_matr = perform_x_val(df, clf_mod, features=features, standardize=True)
            timeafter = TimeUtils.get_time()
            time_diff = timeafter - timebefore
            if all == 1:
                results_file = "variance_thresholds_wo_user_svm"
            else:
                results_file = "variance_thresholds_with_user_svm"

            # save results
            model_name = type(clf_mod).__name__

            conf = dict()
            conf['clf'] = model_name
            conf['f1'] = f1
            conf['threshold'] = thresh
            conf['conf_matr'] = conf_matr_to_list(conf_matr)
            conf['time_diff'] = str(time_diff)
            conf['features'] = str(after)
            conf['num_features'] = len(after)
            conf['all'] = all
            save_result(conf, filename=results_file)
            # increase threshold
            thresh += 0.001
            if f1_prev > f1:
                break
            f1_prev = f1
        gc.collect()


def min_max_scale(data, min_max_scaler):
    cols = data.columns
    x = data.values
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)
    data.columns = cols
    return data


def undo_min_max_scale(data, min_max_scaler):
    cols = data.columns
    x = data.values
    x_scaled = min_max_scaler.inverse_transform(x)
    data = pd.DataFrame(x_scaled)
    data.columns = cols
    return data


def evaluate_best_mi(all):
    """
    evaluates best MI features selection result on full dataset
    :param all: 
    :return: 
    """
    clfs = ['svm', 'nn', 'xgb', 'rf', 'dt', 'nb']

    for clf_name in clfs:
        res = read_feature_mi_results(clf_name, all, sample=True)[0]
        features = res['features']

        data = get_dataset(clf_name)
        clf = get_base_learners(clf_name)

        timebefore = TimeUtils.get_time()
        f1, conf_matr = perform_x_val(data, clf, features, standardize=True)
        timeafter = TimeUtils.get_time()
        time_diff = timeafter - timebefore

        res['f1'] = f1
        res['conf_matr'] = conf_matr_to_list(conf_matr)
        res['time_diff'] = str(time_diff)

        filename = ''
        if all == 1:
            filename = '{}_mi_wo_user_on_full_set'.format(clf_name)
        elif all == 0:
            filename = '{}_mi_with_user_on_full_set'.format(clf_name)
        save_result(res, filename)


def perform_feature_selection_mutual_information(all, sample):
    """
    performs a feature selection based on the k best features according to a chi square test
    :param all: 
    :return: 
    """
    clfs = ['svm', 'nn', 'xgb', 'rf', 'dt', 'nb']

    for clf in clfs:
        data = get_dataset(clf)
        data = data.reset_index(drop=True)

        file = 'results/mutual_information_{}.json'.format(clf)
        if os.path.isfile(file):
            print('Load mutual information from file')
            with open(file) as json_data:
                d = json.load(json_data)
                mi = [(feat[0],feat[1]) for feat in d]
        elif clf == 'xgb' and os.path.isfile('results/mutual_information_svm.json'):
            with open('results/mutual_information_svm.json') as json_data:
                d = json.load(json_data)
                mi = [(feat[0],feat[1]) for feat in d]
        elif clf=='svm' and os.path.isfile('results/mutual_information_xgb.json'):
            with open('results/mutual_information_xgb.json') as json_data:
                d = json.load(json_data)
                mi = [(feat[0],feat[1]) for feat in d]
        else:
            all_features = get_feature_selection(data, all=0)
            X = data[all_features]
            y = data[get_label()]

            print('Calculate mutual information')
            mi_res = mutual_info_classif(X, y)
            mi = [(feat, m) for feat, m in zip(all_features, mi_res)]
            mi.sort(key=lambda tup: tup[1], reverse=True)
            with open(file, 'w') as file:
                json.dump(mi, file)

        features = get_feature_selection(data, all=all)
        mi = [i for i in mi if i[0] in features]
        mi.sort(key=lambda tup: tup[1], reverse=True)

        if all == 1 and 'user__' in mi:
            print("user feature in MI list and all=1 !!!!!")

        if sample:
            indices = load_data_from_CSV(get_root_directory() + '/data/sample/sample_indices_{}.csv'.format(100000))
            data = pd.concat([data, indices], axis=1, join='inner')
            old_f1 = evalute_mutual_info_iteration(data, features, clf, all)
        else:
            old_f1 = get_baseline_result(clf, all)
        interval = 10
        k = len(features)-interval
        while True:

            mi_new = mi[:k]
            k_best_feats = [feat[0] for feat in mi_new]
            print("Number of features to use: {}".format(len(k_best_feats)))

            f1 = evalute_mutual_info_iteration(data, k_best_feats, clf, all)

            if f1 >= old_f1:
                old_f1 = f1
                k -= interval
            else:
                if interval > 1:
                    print('F1 decreased -> set n=1')
                    k += interval
                    interval = 1
                    # k = current best k -1
                    k -= interval
                else:
                    print('F1 decreased and n=1 -> stop.')
                    break
        gc.collect()


def evalute_mutual_info_iteration(df, features, clf, all):
    clf_mod = get_base_learners(name=clf)

    timebefore = TimeUtils.get_time()
    f1, conf_matr = perform_x_val(df, clf_mod, features=features, standardize=True)
    timeafter = TimeUtils.get_time()
    time_diff = timeafter - timebefore
    if all == 1:
        results_file = "{}_mi_wo_user_svm".format(clf)
    else:
        results_file = "{}_mi_with_user_svm".format(clf)

    # save results
    conf = dict()
    conf['clf'] = type(clf_mod).__name__
    conf['f1'] = f1
    conf['conf_matr'] = conf_matr_to_list(conf_matr)
    conf['time_diff'] = str(time_diff)
    conf['features'] = features
    conf['num_features'] = len(features)
    conf['all'] = all
    save_result(conf, filename=results_file)
    return f1


if __name__ == "__main__":
    # perform_feature_selection_variance_threshold(all=1)
    # perform_feature_selection_variance_threshold(all=0)

    # perform_feature_selection_chi_sqaured(all=1)
    # perform_feature_selection_chi_sqaured(all=0)

    # perform_feature_selection_mutual_information(all=0, sample=True)
    # perform_feature_selection_mutual_information(all=1, sample=True)

    # evaluate_best_mi(all=1)
    evaluate_best_mi(all=0)