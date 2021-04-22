import json

import pandas as pd

from DatasetUtils.SampleCreator import get_sample
from FeatureEngineering.FeatureSelector import get_feature_selection
from Learning.LearningUtils import get_dataset, get_testset
from Preprocessing import Preprocessor
from Utility.CSVUtils import load_data_from_CSV, save_df_as_csv
from Utility.Util import get_root_directory


def join_label_and_group(data):
    """
    joins tweet__fake and user__id
    :param data: 
    :return: 
    """
    data_to_join = load_data_from_CSV(get_root_directory()+'/data/data_set_tweet_user_features.csv')
    data_to_join = data_to_join.reset_index(drop=True)
    data = data.reset_index(drop=True)
    data_to_join = data_to_join.reset_index(drop=True)

    data_to_join = data_to_join[['tweet__fake', 'user__id']]
    data = pd.concat([data_to_join, data], axis=1)
    print("Shape after join: {}".format(data.shape))
    return data


def load_user_features(testset):
    """
    loads user features from database and saves them
    :return: 
    """
    from Database.DatabaseHandler import DatabaseHandler
    data = DatabaseHandler.get_user_features_df(testset)
    print(data.shape)
    print(data.head())
    if testset:
        save_df_as_csv(data, "users_testset.csv")
    else:
        save_df_as_csv(data, "users_12_07.csv")


def join_users(testset):
    """
    joins a csv file with the users on user ids
    :param testset: 
    :return: 
    """
    if testset:
        user_data = load_data_from_CSV('../FeatureEngineering/users.csv')
        tweet_data = load_data_from_CSV('../FeatureEngineering/data_set_sample_features_final.csv')
    else:
        user_data = load_data_from_CSV('../FeatureEngineering/users.csv')
        tweet_data = load_data_from_CSV('../FeatureEngineering/data_set_sample_features_final.csv')

    cols_to_keep = list()
    for col in list(tweet_data.columns):
        if "tweet__" in col:
            cols_to_keep.append(col)

    tweet_data = tweet_data[cols_to_keep]

    print(tweet_data.shape)
    print(user_data.shape)

    data = pd.merge(tweet_data, user_data, how='left',  left_on=['tweet__user_id'], right_on=['user__id'])
    if testset:
        save_df_as_csv(data, "testdata/testset_tweet_user_features.csv")
    else:
        save_df_as_csv(data, "data_set_tweet_user_features.csv")


def append_feature(feature_to_append, algo):
    """
    appends a feature to a dataset
    :param feature_to_append: 
    :param algo: 
    :return: 
    """
    data = get_dataset(algo)
    data = data.reset_index(drop=True)

    feature = load_data_from_CSV("../data/data_set_tweet_user_features.csv")[[feature_to_append]]

    data = pd.concat([data, feature], 1)
    return data


def append_feature_from_db(feature_to_append):
    """
    appends a feature that was stored in the DB to the specified filename
    :param feature_to_append: 
    :return: 
    """
    from Database.DatabaseHandler import DatabaseHandler

    data = load_data_from_CSV("data_set_tweet_user_features.csv")
    print("drop " + feature_to_append)
    data = data.drop(feature_to_append, 1)
    print("load " + feature_to_append)
    data['tweet__'+feature_to_append] = data['tweet__id'].map(lambda x: DatabaseHandler.get_feature_by_tweet_id(feature_to_append, x))
    save_df_as_csv(data, "data_set_tweet_user_features.csv")


def concat_dfs(data, data_to_join):
    data_to_join = data_to_join.drop('tweet__id', 1)

    data = pd.concat([data, data_to_join], axis=1)
    print(data.shape)
    for col in data.columns:
        print(col + " NaN: " + str(data[col].isnull().values.any()))
    return data


def combine_data_sets():
    """
    combines doc2vec, topic model, tweet and user feature vectors
    :param norm: 
    :return: 
    """
    text_model_vector_dir = get_root_directory()+"/data/text_data/"
    topic_vector_dir = get_root_directory()+"/data/topics_data/"
    datasets = get_root_directory()+"/data/"

    doc2vec = dict()
    topics = dict()

    doc2vec['nb'] = "d2v_models_300_0_20.csv"
    # doc2vec['dt'] = "d2v_models_200_0_20.csv"
    doc2vec['dt'] = "unigram_bow.csv"
    doc2vec['svm'] = "d2v_models_300_0_20.csv"
    doc2vec['nn'] = "d2v_models_100_0_20.csv"
    doc2vec['xgb'] = "d2v_models_300_0_20.csv"
    # doc2vec['rf'] = "data_set_300_0_20_d2v.csv"
    doc2vec['rf'] = "unigram_bow.csv"


    topics['nb'] = "data_topics_170.csv"
    topics['dt'] = "data_topics_170.csv"
    topics['svm'] = "data_topics_90.csv"
    topics['nn'] = "data_topics_190.csv"
    topics['xgb'] = "data_topics_90.csv"
    topics['rf'] = "data_topics_200.csv"

    configs = ['nb','dt','nn','svm','xgb','rf']
    for conf in configs:
        text_model_vector = load_data_from_CSV(text_model_vector_dir+doc2vec[conf])
        text_model_vector = text_model_vector.reset_index(drop=True)

        tm_cols = [col for col in text_model_vector if 'tweet__id' not in col and 'user__id' not in col and 'tweet__fake' not in col]
        text_model_vector = text_model_vector[tm_cols]

        topic_vector = load_data_from_CSV(topic_vector_dir+topics[conf])
        topic_vector = topic_vector.reset_index(drop=True)

        data = load_data_from_CSV(get_root_directory()+"/data/data_set_tweet_user_features.csv")
        data = Preprocessor.replace_missing_possibly_sensitive(data)
        features = get_feature_selection(data)
        features.extend(['tweet__fake', 'user__id', 'tweet__id'])
        data = data[features]
        data = data.reset_index(drop=True)

        data = pd.concat([data, text_model_vector], axis=1)
        data = pd.concat([data, topic_vector], axis=1)

        print(data.shape)
        save_df_as_csv(data, datasets+'data_set_'+conf+'.csv')

def combine_testsets(testset_only=False):
    """
    combines doc2vec, topic model, tweet and user feature vectors to build the testset. 
    Does not include tweets that are shifted from the training data.
    :param testset_only: True if LDA and Doc2Vec models trained on testdata shoud be used
    :return: 
    """
    text_model_vector_dir = get_root_directory()+"/data/text_data/"
    topic_vector_dir = get_root_directory()+"/data/topics_data/"
    datasets = get_root_directory()+"/data/testdata/"

    doc2vec = dict()
    topics = dict()

    if testset_only:
        doc2vec['nb'] = "d2v_models_testset_300_0_20.csv"
        doc2vec['dt'] = "testset_only_unigram_bow.csv"
        doc2vec['svm'] = "d2v_models_testset_300_0_20.csv"
        doc2vec['nn'] = "d2v_models_testset_100_0_20.csv"
        doc2vec['xgb'] = "d2v_models_testset_300_0_20.csv"
        doc2vec['rf'] = "testset_only_unigram_bow.csv"

        topics['nb'] = "data_testset_topics_170.csv"
        topics['dt'] = "data_testset_topics_170.csv"
        topics['svm'] = "data_testset_topics_90.csv"
        topics['nn'] = "data_testset_topics_190.csv"
        topics['xgb'] = "data_testset_topics_90.csv"
        topics['rf'] = "data_testset_topics_200.csv"
    else:
        doc2vec['nb'] = "testset_d2v_300.csv"
        doc2vec['dt'] = "testset_unigram_bow.csv"
        doc2vec['svm'] = "testset_d2v_300.csv"
        doc2vec['nn'] = "testset_d2v_100.csv"
        doc2vec['xgb'] = "testset_d2v_300.csv"
        doc2vec['rf'] = "testset_unigram_bow.csv"

    clfs = ['nb','dt','nn','svm','xgb','rf']

    for clf in clfs:
        text_model_vector = load_data_from_CSV(text_model_vector_dir+doc2vec[clf])
        if testset_only:
            topic_vector = load_data_from_CSV(topic_vector_dir+topics[clf])
            data = load_data_from_CSV(get_root_directory()+"/data/testdata/testset_tweet_user_features_complete.csv")
        else:
            topic_vector = load_data_from_CSV(topic_vector_dir+'testset_topics_'+clf+'.csv')
            data = load_data_from_CSV(get_root_directory()+"/data/testdata/testset_tweet_user_features.csv")
        data = Preprocessor.replace_missing_possibly_sensitive(data)
        features = get_feature_selection(data)
        features.extend(['tweet__fake', 'user__id', 'tweet__id'])
        data = data[features]

        data = pd.concat([data, text_model_vector], axis=1)

        # for testset only topics won't be used, because no topics (with large number of topics) could be infered
        if not testset_only:
            data = pd.concat([data, topic_vector], axis=1)

        # print(data.index)
        print(data.shape)
        if testset_only:
            save_df_as_csv(data, datasets+'testset_only_'+clf+'.csv')
        else:
            save_df_as_csv(data, datasets+'testset_'+clf+'.csv')


def get_real_news_to_include():
    """
    :return: ids that shifted into the testset 
    """
    with open(get_root_directory()+'/data/testdata/ids_include_in_testset.json') as json_data:
        d = json.load(json_data)
    return d


def append_all_attributes_not_in_data(data, data_with_features):
    """
    Appends all attributes from 'data_with_features' which are not contained in 'data' to 'data'.
    Resets index!
    :param data: data to append
    :param data_with_features: data with additional attributes
    :return: 
    """
    data = data.reset_index(drop=True)
    print("data: {}".format(data.shape))

    data_with_features = data.reset_index(drop=True)
    print("data_with_features: {}".format(data.shape))

    count = 0
    for col in data_with_features.columns:
        if col not in data_with_features:
            data[col] = data_with_features[col]
            count += 1

    print("{} attributes appended.".format(count))
    print("new data: {}".format(data.shape))
    return data

def build_testset(all, clf_name, keep_all_features=False):
    if clf_name:
        train = get_dataset(clf_name)
        test = get_testset(clf_name)
    else:
        train = load_data_from_CSV(get_root_directory() + "/data/data_set_tweet_user_features.csv")
        test = load_data_from_CSV(get_root_directory() + "/data/testdata/testset_tweet_user_features.csv")

    features_to_extend = ['tweet__id', 'user__id', 'tweet__fake']
    if keep_all_features:
        tmp_features = [col for col in train.columns]
    else:
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

if __name__ == "__main__":

    # load_user_features()
    # join_users()

    # combine datasets (tweet/user features, Doc2Vec/BOW, topics) to a dataset for each learner
    # combine_data_sets()

    # combine datasets (tweet/user features, Doc2Vec/BOW, topics) to a testset for each learner
    combine_testsets(testset_only=True)

    # create testset
    # data = build_testset(0, clf_name=None)
    # save_df_as_csv(data, '../data/testdata/testset_tweet_user_features_complete.csv')
