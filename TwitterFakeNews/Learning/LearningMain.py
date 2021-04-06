import gc
import pandas as pd
from pandas import np
import os

from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from DatasetUtils.DataSetCombiner import get_real_news_to_include, join_label_and_group
from DatasetUtils.Standardizer import standardize_data
from FeatureEngineering.FeatureSelector import get_label, \
    get_feature_selection
from Learning.EvaluationMetrics import print_result, get_result
from Learning.LearningUtils import get_base_learners, get_dataset, save_result, \
    get_learner_names, get_testset, get_learner_and_features
from Learning.XValidation import XValidation
from Utility.CSVUtils import load_data_from_CSV, save_df_as_csv
from Utility.TimeUtils import TimeUtils


def perform_x_val(data, clf, features, standardize=False):
    """
    performs a X-Validation
    :param data: data to evaluate
    :param clf: classifier to use
    :param features: features to use
    :param standardize: whether the data should be standardized for SVM and Neural Network
    :return: 
    """

    timebefore = TimeUtils.get_time()

    print(data.shape)

    xVal = XValidation(features)
    xVal.groupKFold(data, clf, n_splits=10, standardize=standardize)
    acc = xVal.get_accuracy()
    p = xVal.get_precision()
    r = xVal.get_recall()
    f1 = xVal.get_F1_score()

    print(xVal.conf_matr)
    print("Accuracy: " + str(acc))
    print("Precision: " + str(p))
    print("Recall: " + str(r))
    print("F1: " + str(f1))

    timeafter = TimeUtils.get_time()
    diff = timeafter - timebefore
    print("Duration: " + str(diff))
    return f1, xVal.conf_matr


def evaluate_on_holdout_set(data_train, data_test, clf, features, all, predict_proba=False, res_file=None):
    """
    evaluates a learner on a holdout set
    :param data_train: training data
    :param data_test: holdout set
    :param clf: classifier
    :param features: features
    :param all: use case
    :param predict_proba: if true, uses predict_proba instead of predict to get the class probabilities
    :param res_file: if given, saves the results in dataframe 
    :return: 
    """

    conf_matr = np.zeros((2, 2))

    if all == 2:
        features = [feat for feat in features if 'user__' in feat]

    X_train = data_train[features].as_matrix()
    print(X_train.shape)
    y_train = data_train[get_label()].as_matrix()

    X_test = data_test[features].as_matrix()
    y_test = data_test[get_label()].as_matrix()

    if isinstance(clf, GaussianNB) or isinstance(clf, MLPClassifier) or isinstance(clf, LinearSVC) or isinstance(clf,
                                                                                                                 Pipeline):
        print("Standardize Data")
        X_train, X_test = standardize_data(X_train, X_test)

    clf.fit(X_train, y_train)

    if predict_proba:
        if isinstance(clf, LinearSVC) or isinstance(clf, Pipeline):
            y_pred = clf.predict(X_test)
        else:
            y_pred = clf.predict_proba(X_test)

        if len(y_pred.shape) == 2:
            res_df = pd.DataFrame({"{}_{}".format(type(clf).__name__, 0): list(y_pred[:, 0])})
            res_df["{}_{}".format(type(clf).__name__, 1)] = list(y_pred[:, 1])
        elif len(y_pred.shape) == 1:
            # probability of class 0/1 is 1 or 0, repsectively (=one-hot-encoding)
            y_pred_0 = [int(pred == 0) for pred in y_pred]
            y_pred_1 = [int(pred == 1) for pred in y_pred]
            res_df = pd.DataFrame({"{}_{}".format(type(clf).__name__, 0): y_pred_0})
            res_df["{}_{}".format(type(clf).__name__, 1)] = list(y_pred_1)

        print("predict_proba done for {}".format(type(clf).__name__))
    else:
        y_pred = clf.predict(X_test)

        conf_matr = np.add(np.array(confusion_matrix(y_test, y_pred, [1, 0])), conf_matr)
        print(type(clf).__name__)
        print_result(conf_matr)
        res = get_result(conf_matr)
        res['clf'] = type(clf).__name__
        save_result(res, 'results/results_testset_{}'.format(all))
        joblib.dump(clf, 'models/{}_for_test_{}.pkl'.format(res['clf'], all))

        res_df = pd.DataFrame({type(clf).__name__: list(y_pred)})

    if res_file is not None:
        prev_res_df = load_data_from_CSV(res_file)
        if type(clf).__name__ in prev_res_df:
            prev_res_df = prev_res_df.drop(type(clf).__name__, 1)
        res_df = pd.concat([prev_res_df, res_df], axis=1)
        save_df_as_csv(res_df, res_file)


def build_model(data, features, clf):
    label = get_label()
    X = data[features]
    y = data[label]

    clf.fit(X, y)


def get_baselines(all):
    """
    evaluates all algorithms for one of the use cases 
    :param all: 1: tweet features, 0: tweet+user features
    :return: 
    """
    clfs = ['nb', 'dt', 'nn', 'svm', 'xgb', 'rf']

    for clf_name in clfs:
        data = get_dataset(clf_name)
        features = get_feature_selection(data=data, all=all)

        clf = get_base_learners(name=clf_name)
        f1, conf_matr = perform_x_val(data, clf, features=features)

def build_testset(all, clf_name):
    train = get_dataset(clf_name)
    test = get_testset(clf_name)

    features_to_extend = ['tweet__id', 'user__id', 'tweet__fake']
    tmp_features = get_feature_selection(train, all)
    tmp_features.extend(features_to_extend)
    train = train[tmp_features]
    test = test[tmp_features]

    ids = get_real_news_to_include()
    to_shift = train[train['tweet__id'].isin(ids)]
    train = train[~train['tweet__id'].isin(ids)]
    print("Shape train: {}".format(train.shape))

    test = test.append(to_shift)
    test = test.reset_index(drop=True)
    print("Shape test: {}".format(test.shape))
    return test

def evaluate_testset(all, clfs=None, predict_proba=False):
    """
    evaluates the testset(s)
    :param all: 1: tweet features, 0: tweet+user features 
    :param clfs: list of clf names to evaluate, if None then all learners are evaluated
    :param predict_proba: set to true, if the result should serve as input for weighted voting
    :return: 
    """
    if clfs is None:
        clfs = get_learner_names()

    for clf_name in clfs:
        if all == 2:
            tmp_all = 0
        else:
            tmp_all = all

        clf, features = get_learner_and_features(clf_name, all=tmp_all)
        train = get_dataset(clf_name)
        test = get_testset(clf_name)

        # shift tweets into testset
        features_to_extend = ['tweet__id', 'user__id', 'tweet__fake']
        tmp_features = get_feature_selection(train, all)
        tmp_features.extend(features_to_extend)
        train = train[tmp_features]
        test = test[tmp_features]

        ids = get_real_news_to_include()
        to_shift = train[train['tweet__id'].isin(ids)]
        train = train[~train['tweet__id'].isin(ids)]
        print("Shape train: {}".format(train.shape))

        test = test.append(to_shift)
        test = test.reset_index(drop=True)
        print("Shape test: {}".format(test.shape))

        if predict_proba:
            res_file = 'results/testset_results_weighted_{}.csv'.format(all)
        else:
            res_file = 'results/testset_results_user_only_{}.csv'.format(all)
        if not os.path.isfile(res_file):
            y_tweet_ids = test[['tweet__id']]
            save_df_as_csv(y_tweet_ids, res_file)

        if predict_proba:
            evaluate_on_holdout_set(data_train=train, data_test=test, clf=clf, features=features, all=all, predict_proba=True,
                                res_file=res_file)
        else:
            evaluate_on_holdout_set(data_train=train, data_test=test, clf=clf, features=features, all=all,
                                res_file=res_file)
        train = None
        test = None
        gc.collect()


def evaluate_component(filename, clf_name):
    """
    evaluates a componen
    :param filename: 
    :param clf_name: 
    :return: 
    """
    comp = load_data_from_CSV(filename)
    if 'tweet__fake' not in comp.columns and 'user__id' not in comp.columns:
        comp = join_label_and_group(comp)

    features = [feat for feat in comp.columns if 'tweet__fake' not in feat and 'user__id' not in feat]
    f1, conf_matr = perform_x_val(data=comp, clf=get_base_learners(clf_name), features=features, standardize=False)


def evaluate_best_configurations(clf_name, all):
    """
    evaluate the best configuration of features and parameters
    :param clf_name: name of the classifier
    :param all: 1: tweet features, 0: tweet + user features
    :return: 
    """
    clf, features = get_learner_and_features(clf_name, all=all)
    data = get_dataset(clf_name)
    perform_x_val(data=data, clf=clf, features=features, standardize=True)


def xval_on_testset(clf_name, all):
    """
    performs a 10-fold cross-validation on the testset
    """
    clf = get_base_learners(clf_name)
    data = build_testset(all, clf_name)
    features = get_feature_selection(data, all=all)
    print(features)
    print("Shape of data for training: {}".format(data.shape))
    perform_x_val(data=data, clf=clf, features=features, standardize=True)


if __name__ == "__main__":
    # clf_name: nb = Naïve Bayes, dt = Decision Tree, svm = SVM, nn = Neural Network, xgb = XGBoost, rf = Random Forest
    # Use Case 1: all=1, Use Case 2: all=0

    # evaluate the final results
    evaluate_best_configurations(clf_name='nb', all=1)

    # evaluate the testset
    # evaluate_testset(all=1, clfs=['nb'])
    # evaluate_testset(all=1, predict_proba=False)

    # evaluate with only user features
    # evaluate_testset(all=2, predict_proba=False)

    # evaluate_component('../data/text_data/unigram_bow.csv', 'nb')

    # cross-validate test set
    # xval_on_testset(clf_name='svm', all=1)
