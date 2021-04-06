import json
from collections import Counter
import numpy as np
import itertools

from sklearn.metrics import confusion_matrix
from DatasetUtils.DataSetCombiner import get_real_news_to_include
from FeatureEngineering.FeatureSelector import get_feature_selection
from Learning.EvaluationMetrics import print_result, calc_f1, calc_precision, calc_recall
from Learning.LearningUtils import get_base_learners, get_sklearn_clf_names, \
    get_learner_and_features, get_testset, list_to_conf_matr, get_clf_names
from Learning.Voting import Voting
from Utility.CSVUtils import load_data_from_CSV, read_csv, save_df_as_csv
from Utility.Util import get_root_directory


def get_donald_trump_ids():
    ids = [829681034564341760,
           830751875578355713,
           843090516283723776,
           854268119774367745,
           858659215451271168,
           868047480847568896,
           871328428963901440,
           877101173412638720]
    return ids


def load_testset_predictions(all, weighted=False):
    """
    loads the stored predictions
    :param all: 
    :param weighted: 
    :return: 
    """
    if weighted:
        return load_data_from_CSV('results/testset_results_weighted_{}.csv'.format(all))
    else:
        return load_data_from_CSV('results/testset_results_{}.csv'.format(all))


def get_y_dict():
    tweets = read_csv(get_root_directory() + '/data/testdata/politifact_false.csv')
    tweet_ids_false = [t[0] for t in tweets]
    map = {int(id): 1 for id in tweet_ids_false}

    tweets = read_csv(get_root_directory() + '/data/testdata/politifact_pants_on_fire.csv')
    tweet_ids_pof = [t[0] for t in tweets]
    map.update({int(id): 1 for id in tweet_ids_pof})

    tweets = read_csv(get_root_directory() + '/data/testdata/politifact_true.csv')
    tweet_ids_true_pf = [t[0] for t in tweets]
    map.update({int(id): 0 for id in tweet_ids_true_pf})

    tweet_ids_true_train = get_real_news_to_include()
    map.update({int(id): 0 for id in tweet_ids_true_train})
    return map


def load_fake_ids():
    tweets = read_csv(get_root_directory() + '/data/testdata/politifact_false.csv')
    tweet_ids_false = [t[0] for t in tweets]
    map = {int(id): 'false' for id in tweet_ids_false}

    tweets = read_csv(get_root_directory() + '/data/testdata/politifact_pants_on_fire.csv')
    tweet_ids_pof = [t[0] for t in tweets]
    map.update({int(id): 'pants_on_fire' for id in tweet_ids_pof})
    return map


def load_real_ids():
    tweets = read_csv(get_root_directory() + '/data/testdata/politifact_true.csv')
    tweet_ids_true_pf = [t[0] for t in tweets]
    map = {int(id): 'true_pf' for id in tweet_ids_true_pf}

    tweet_ids_true_train = get_real_news_to_include()
    map.update({int(id): 'true_train' for id in tweet_ids_true_train})
    return map

def load_result(clf_name, all):
    clf = get_sklearn_clf_names(clf_name)
    if clf_name == 'svm':
        clf = 'Pipeline'

    with open('results/results_testset_{}.json'.format(all)) as file:
        results = json.load(file)

    res = ''
    for r in results:
        if r['clf'] == clf:
            res = r
            break

    print("F1: {}".format(res['f1']))
    print("Precision: {}".format(calc_precision(list_to_conf_matr(res['conf_matr']))))
    print("Recall: {}".format(calc_recall(list_to_conf_matr(res['conf_matr']))))
    print(list_to_conf_matr(res['conf_matr']))

def get_id_dict():
    map = load_fake_ids()
    map.update(load_real_ids())
    return map


def vote_results(all, result_df=None, cols=get_sklearn_clf_names()):
    if result_df is None:
        result_df = load_testset_predictions(all)
    y_map = get_y_dict()

    if 'LinearSVC' in cols:
        cols.remove('LinearSVC')
        cols.append('Pipeline')
        y_preds_tmp = result_df[cols]
    else:
        y_preds_tmp = result_df[cols]

    print("Perform voting with {}".format(cols))
    result_df['mean'] = y_preds_tmp.mean(axis=1)
    result_df['vote'] = result_df['mean'].map(lambda x: Voting.determine_class(x))
    result_df['y'] = result_df['tweet__id'].map(lambda x: y_map[x])

    conf_matr = np.array(confusion_matrix(result_df['y'], result_df['vote'], [1, 0]))

    print_result(conf_matr)
    return conf_matr


def weighted_vote_results(all, result_df=None, cols=get_sklearn_clf_names()):
    if result_df is None:
        result_df = load_testset_predictions(all, weighted=True)
    y_map = get_y_dict()

    if 'LinearSVC' in cols:
        cols.remove('LinearSVC')
        cols.append('Pipeline')
    # y_preds_tmp = result_df[cols]

    print("Perform voting with {}".format(cols))

    cols = [col for col in list(result_df.columns) if col[:-2] in cols]

    cols_0 = [col for col in cols if '0' in col]
    cols_1 = [col for col in cols if '1' in col]

    result_df['w_avg_0'] = result_df[cols_0].mean(axis=1)
    result_df['w_avg_1'] = result_df[cols_1].mean(axis=1)
    result_df['vote_res'] = result_df.apply(lambda x: Voting.determine_class_weighted(x['w_avg_0'], x['w_avg_1']),
                                                    axis=1)
    result_df['y'] = result_df['tweet__id'].map(lambda x: y_map[x])

    conf_matr = np.array(confusion_matrix(result_df['y'], result_df['vote_res'], [1, 0]))

    print_result(conf_matr)
    return conf_matr


def find_best_voting_combination(all):
    cols = get_sklearn_clf_names()
    result_df = load_testset_predictions(all)

    f1_best = 0
    best_cols = None
    best_conf = None
    for i in range(2, len(cols)+1):
        combs = itertools.combinations(cols, i)
        for tmp_cols in combs:
            tmp_cols = list(tmp_cols)
            conf_matr = vote_results(result_df=result_df, cols=tmp_cols, all=all)
            f1 = calc_f1(conf_matr)
            # print("{}: {}".format(f1, tmp_cols))
            if f1 > f1_best:
                f1_best = f1
                best_cols = tmp_cols
                best_conf = conf_matr

    print("Best: {}".format(best_cols))
    print_result(best_conf)


def distribution_of_result(clf_name, all, ids=None, print_res=True):
    """
    prints the distribution of the results
    :param clf_name: name of the classifier
    :param all: if 1, only tweet features
    :param ids: if not none, only the ids in 'ids' are considered
    :return: 
    """
    id_map = get_id_dict()
    y_map = get_y_dict()

    if clf_name == 'svm':
        clf_name = 'Pipeline'
    else:
        clf_name = type(get_base_learners(clf_name)).__name__
    y_preds = load_testset_predictions(all)

    distr_count = dict()
    distr_count['TP'] = Counter()
    distr_count['FN'] = Counter()
    distr_count['FP'] = Counter()
    distr_count['TN'] = Counter()

    distr = {key: {source: [] for source in set([v for v in id_map.values()])} for key in ['TP', 'FN', 'FP', 'TN']}

    for index, row in y_preds.iterrows():
        pred = row[clf_name]
        tweet_id = row['tweet__id']

        source = id_map[tweet_id]
        y = y_map[tweet_id]

        if ids is not None and tweet_id not in ids:
            continue

        if y == pred and pred == 1:
            distr_count['TP'][source] += 1
            distr['TP'][source].append(tweet_id)

        elif y != pred and pred == 1:
            distr_count['FP'][source] += 1
            distr['FP'][source].append(tweet_id)

        elif y != pred and pred == 0:
            distr_count['FN'][source] += 1
            distr['FN'][source].append(tweet_id)

        else:
            distr_count['TN'][source] += 1
            distr['TN'][source].append(tweet_id)

    if print_res:
        print(distr_count)
        for key, values in distr.items():
            print('{}:'.format(key))
            for key2, values2 in distr[key].items():
                print('\t{}:'.format(key2))
                print('\t{}'.format(values2))
    return distr_count, distr


def get_most_misclassifier_examples(all):
    fn = []
    fp = []

    for clf in get_clf_names():
        distr_count, distr = distribution_of_result(clf, all, print_res=False)

        fns = distr['FN']
        for value in fns.values():
            fn.extend(value)

        fps = distr['FP']
        for value in fps.values():
            fp.extend(value)

    return Counter(fp), Counter(fn)

def get_most_correct_classified_examples(all):
    tp = []
    tn = []

    for clf in get_clf_names():
        distr_count, distr = distribution_of_result(clf, all, print_res=False)

        tps = distr['TP']
        for value in tps.values():
            tp.extend(value)

        tns = distr['TN']
        for value in tns.values():
            tn.extend(value)

    return Counter(tp), Counter(tn)


def store_ids_in_df(ids, all):
    testset = get_testset('nn')
    features = get_feature_selection(testset, all)
    features.append('user__id')
    features.append('tweet__id')
    features.append('tweet__fake')
    cols = [col for col in features if 'd2v' not in col and 'tweet__topic' not in col]

    testset = testset[cols]
    testset = testset[testset['tweet__id'].isin(ids)]

    save_df_as_csv(testset, 'tweets_most_tps.csv')


def store_ids(ids, clf_name, all, filename):
    testset = get_testset(clf_name)
    clf, features = get_learner_and_features(clf_name, all=all)
    features.append('user__id')
    features.append('tweet__id')
    features.append('tweet__fake')

    testset = testset[features]
    testset = testset[testset['tweet__id'].isin(ids)]

    save_df_as_csv(testset, filename)


def compare_0_and_2(clf_name, print_dist=True):
    """
    counts how many times a tweet was classified in one of the classes with either user+tweet (0) or user features (2)
    :param clf_name: 
    :param print_dist: 
    :return: 
    """
    count0, dist0 = distribution_of_result(clf_name, all=0, print_res=False)
    count2, dist2 = distribution_of_result(clf_name, all=2, print_res=False)

    tp0 = list()
    for value in dist0['TP'].values():
        tp0.extend(value)

    tn0 = list()
    for value in dist0['TN'].values():
        tn0.extend(value)

    fp0 = list()
    for value in dist0['FP'].values():
        fp0.extend(value)

    fn0 = list()
    for value in dist0['FN'].values():
        fn0.extend(value)

    tp2 = list()
    for value in dist2['TP'].values():
        tp2.extend(value)

    tn2 = list()
    for value in dist2['TN'].values():
        tn2.extend(value)

    fp2 = list()
    for value in dist2['FP'].values():
        fp2.extend(value)

    fn2 = list()
    for value in dist2['FN'].values():
        fn2.extend(value)

    tp = list()
    tp.extend(tp0)
    tp.extend(tp2)
    tn = list()
    tn.extend(tn0)
    tn.extend(tn2)
    fp = list()
    fp.extend(fp0)
    fp.extend(fp2)
    fn = list()
    fn.extend(fn0)
    fn.extend(fn2)

    tp_counter = Counter(tp)
    tn_counter = Counter(tn)
    fp_counter = Counter(fp)
    fn_counter = Counter(fn)

    if print_dist:
        print("TP: {}".format(tp_counter))
        print("TN: {}".format(tn_counter))
        print("FP: {}".format(fp_counter))
        print("FN: {}".format(fn_counter))

    return tp_counter, tn_counter, fp_counter, fn_counter


if __name__ == '__main__':

    # perform voting on the testset
    # requires to evaluate the testset for each learner first in LearningMain!
    vote_results(all=1, cols=[get_sklearn_clf_names('dt'), get_sklearn_clf_names('nn'), get_sklearn_clf_names('xgb')])
    # vote_results(all=0, cols=[get_sklearn_clf_names('dt'), get_sklearn_clf_names('nn'), get_sklearn_clf_names('svm'), get_sklearn_clf_names('xgb'), get_sklearn_clf_names('rf')])
    # weighted_vote_results(all=1, cols= [get_sklearn_clf_names('nn'), get_sklearn_clf_names('xgb')])
    # weighted_vote_results(all=0, cols=[get_sklearn_clf_names('dt'), get_sklearn_clf_names('nn'), get_sklearn_clf_names('svm'), get_sklearn_clf_names('xgb'), get_sklearn_clf_names('rf')])

    # fps, fns = get_most_misclassifier_examples(all)
    # tps, tns = get_most_correct_classified_examples(all)

    # count, distr = distribution_of_result('rf', all=2)
    # count, distr = distribution_of_result('rf', all=1, ids=get_donald_trump_ids())
    # compare_0_and_2('svm')

