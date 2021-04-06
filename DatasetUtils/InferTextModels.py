import json
import re

import pandas as pd
import numpy as np

from NLP.NLPUtils import NLPUtils
from TextRepresentation.Doc2Vec import Doc2Vec
from TextRepresentation.TextModel import TextModel
from Utility.CSVUtils import load_data_from_CSV, save_df_as_csv
from Utility.Util import get_root_directory


class InferTextModels:
    def __init__(self, test_data):
        self.test_data = test_data

    def build_doc2vec(self, size):
        """
        builds Doc2Vec dataset
        requires: doc2vec model
        :return: 
        """
        d2v = Doc2Vec(X=None, y=None)
        d2v.load_model(self.get_text_model_by_size(size))

        texts = self.test_data['tweet__additional_preprocessed_text'].tolist()
        nr_train_instances = len(texts)
        test_arrays = np.zeros((nr_train_instances, d2v.model.vector_size))
        for i in range(len(texts)):
            test_arrays[i] = d2v.model.infer_vector(NLPUtils.str_list_to_list(texts[i]))

        doc2vec_data = pd.DataFrame(data=test_arrays, columns=["tweet__d2v_{}".format(i) for i in range(d2v.model.vector_size)])
        return doc2vec_data

    def build_topics(self, clf_name):
        """
        builds the topics 
        requires: dict, tf-idf model (without stopwords), lda model
        :return: 
        """
        t = TextModel()

        tweets = [[token for token in NLPUtils.str_list_to_list(tweet) if
                    token != "USERMENTION" and token != "URL"] for tweet
                    in self.test_data['tweet__additional_preprocessed_wo_stopwords'].tolist()]

        t.num_topics = self.get_topic_model_by_clf_name(clf_name)

        # load required models
        t.load_dict_topics(t.num_topics)
        t.load_tf_idf_topic_model(t.num_topics)
        t.load_lda_model(t.num_topics)

        doc_bow = [t.dictionary.doc2bow(tweet) for tweet in tweets]
        corpus_tfidf_test = t.tf_idf[doc_bow]
        t.corpus_lda = t.lda[corpus_tfidf_test]
        return t.get_topics_as_df()

    @staticmethod
    def init_tfidf_model():
        """
        initializes the tf-idf model for 'tf_idf_sum'
        :return: 
        """
        data = load_data_from_CSV(get_root_directory()+"/data/data_set_tweet_user_features.csv")

        t = TextModel()
        tweets = [[token for token in NLPUtils.str_list_to_list(tweet) if
                    token != "USERMENTION" and token != "URL" and not re.match('^\d+$', token)] for tweet
                    in data['tweet__additional_preprocessed_wo_stopwords'].tolist()]
        t.init_corpus(tweets)
        t.calc_tf_idf()
        t.save_dict()
        t.save_tf_idf_model()


    def append_tf_idf_sum(self):
        """
        appends the tweet feature 'tweet__tf_idf_sum'
        requires: dict, tf-idf model (without stopwords)
        :return: 
        """
        t = TextModel()
        t.load_dict()
        t.load_tf_idf_model()

        tweets = [[token for token in NLPUtils.str_list_to_list(tweet) if
                    token != "USERMENTION" and token != "URL" and not re.match('^\d+$', token)] for tweet
                    in self.test_data['tweet__additional_preprocessed_wo_stopwords'].tolist()]
        doc_bow = [t.dictionary.doc2bow(tweet) for tweet in tweets]
        t.corpus_tfidf = t.tf_idf[doc_bow]
        self.test_data['tweet__tf_idf_sum'] = t.get_tf_idf_series()

    def append_bigram_tf_idf_sum(self):
        """
        appends the tweet feature 'tweet__tf_idf_sum'
        requires: dict, tf-idf model (without stopwords)
        :return: 
        """
        t = TextModel()
        t.load_bigram_dict()
        t.load_tf_idf_bigram_model()

        tmp = self.test_data['tweet__additional_preprocessed_text'].tolist()
        bigram_tweets = [NLPUtils.generate_n_grams(NLPUtils.str_list_to_list(tweet), 2) for tweet in tmp]

        doc_bow = [t.dictionary.doc2bow(tweet) for tweet in bigram_tweets]
        t.corpus_tfidf = t.tf_idf[doc_bow]
        self.test_data['tweet__bigram_tf_idf_sum'] = t.get_tf_idf_series()

    def append_pos_trigrams(self):
        """
        infers POS trigrams for the testset
        requires: tf-idf BOW model, BOW dict, BOW filtered dict, VocabTransform model
        :return: 
        """
        tweets_tags = self.test_data['tweet__pos_tags'].tolist()

        trigram_pos = []
        for tweet in tweets_tags:
            tweet = json.loads(tweet)
            pos_tags = [token['tag'] for token in tweet]
            trigram_pos.append(NLPUtils.generate_n_grams(pos_tags, 3))

        t = TextModel()
        t.load_tf_idf_bow_model(variant='pos_trigram')
        t.load_bow_dict(variant='pos_trigram')
        t.load_bow_filtered_dict(variant='pos_trigram')
        t.load_vt_model(variant='pos_trigram')

        doc_bow = [t.dictionary.doc2bow(tweet) for tweet in trigram_pos]
        corpus = t.tf_idf[doc_bow]
        corpus = t.vt[corpus]

        trigram_vectors = t.build_bag_of_words_df(corpus)
        map = dict()
        for key, vector in trigram_vectors.items():
            map['tweet__contains_pos_trigram_{}'.format(re.sub(" ", "_", str(key)))] = vector

        # ensure same order as in training data
        train_cols = [col for col in load_data_from_CSV(get_root_directory()+"/data/data_set_tweet_user_features.csv").columns if 'pos_trigram' in col]
        for col_name in train_cols:
            self.test_data[col_name] = map[col_name]

    def build_bag_of_words_set(self):
        """
        infers bag-of-words for the testset
        requires: unigram tf-idf BOW model, unigram BOW dict, unigram BOW filtered dict, unigram VocabTransform model
        :return: 
        """
        tweets = [NLPUtils.str_list_to_list(tweet) for tweet in
                  self.test_data['tweet__additional_preprocessed_wo_stopwords'].tolist()]
        for i in range(len(tweets)):
            tweets[i] = [token for token in tweets[i] if token != 'URL' and token != 'USERMENTION' and not re.match('^\d+$',token)]

        t = TextModel()
        t.load_tf_idf_bow_model(variant='unigram')
        t.load_bow_dict(variant='unigram')
        t.load_bow_filtered_dict(variant='unigram')
        t.load_vt_model(variant='unigram')

        doc_bow = [t.dictionary.doc2bow(tweet) for tweet in tweets]
        corpus = t.tf_idf[doc_bow]
        corpus = t.vt[corpus]

        bow_vectors = t.build_bag_of_words_df(corpus)
        map = dict()
        for key, vector in bow_vectors.items():
            map['tweet__contains_{}'.format(key)] = vector

        # ensure same order as in training data
        text_vectors_dir = get_root_directory()+"/data/text_data/"
        train_cols = [col for col in load_data_from_CSV(text_vectors_dir + "unigram_bow.csv").columns if
                      'user__id' not in col and 'tweet__fake' not in col]

        df = pd.DataFrame(index=self.test_data.index)
        for col_name in train_cols:
            df[col_name] = map[col_name]

        return df

    @staticmethod
    def get_text_model_by_size(size):
        """
        returns the text model for a given classifier
        :return: 
        """
        doc2vec = dict()
        folder = get_root_directory()+"/data/d2v_models/"
        doc2vec[100] = folder+"doc2vec_model_100_0_20.d2v"
        doc2vec[200] = folder+"doc2vec_model_200_0_20.d2v"
        doc2vec[300] = folder+"doc2vec_model_300_0_20.d2v"
        return doc2vec[size]

    @staticmethod
    def get_topic_model_by_clf_name(clf_name):
        """
        returns the topic model for a given classifier
        :return: 
        """
        topics = dict()
        topics['nb'] = 170
        topics['dt'] = 170
        topics['svm'] = 90
        topics['nn'] = 190
        topics['xgb'] = 90
        topics['rf'] = 200
        return topics[clf_name]


def append_tf_idfs_and_store():
    """
    appends 'tweet__tf_idf_sum' and 'tweet__bigram_tf_idf_sum' and 'pos_trigrams' BOW to the testset
    :return: 
    """
    data = load_data_from_CSV(get_root_directory() + "/FeatureEngineering/testset_tweet_user_features.csv")
    print(data.shape)

    tb = InferTextModels(data)
    print('append tf_idf_sum')
    tb.append_tf_idf_sum()
    print('append_bigram_tf_idf_sum')
    tb.append_bigram_tf_idf_sum()
    print('append_pos_trigrams')
    tb.append_pos_trigrams()

    print(tb.test_data.shape)
    save_df_as_csv(tb.test_data, get_root_directory() + "/data/testdata/testset_tweet_user_features.csv")


def create_topics_sets():
    """
    creates the topics data sets for the testset
    :return: 
    """
    data = load_data_from_CSV(get_root_directory() + "/data/testset_tweet_user_features.csv")
    print(data.shape)

    for clf_name in ['nb','dt','svm','nn','xgb','rf']:
        tb = InferTextModels(data)
        topic_data = tb.build_topics(clf_name)
        folder = get_root_directory()+"/data/topics_data/"
        save_df_as_csv(topic_data, folder+'testset_topics_'+clf_name+".csv")


def create_d2v_sets():
    """
    creates the Doc2Vec datasets for the testset
    :return: 
    """
    data = load_data_from_CSV(get_root_directory() + "/data/testset_tweet_user_features.csv")
    print(data.shape)

    for i in [100,200,300]:
        tb = InferTextModels(data)
        topic_data = tb.build_doc2vec(i)
        folder = get_root_directory()+"/data/text_data/"
        save_df_as_csv(topic_data, folder+'testset_d2v_'+str(i)+".csv")

def create_bow_set():
    """
    creates the Bag-of-words dataset for the testset
    :return: 
    """
    data = load_data_from_CSV(get_root_directory() + "/data/testset_tweet_user_features.csv")
    print(data.shape)

    tb = InferTextModels(data)
    bow_data = tb.build_bag_of_words_set()
    # folder = '../data/testdata/'
    folder = get_root_directory()+"/data/text_data/"
    save_df_as_csv(bow_data, folder+'testset_unigram_bow.csv')



if __name__ == '__main__':
    # append_tf_idfs_and_store()
    # create_topics_sets()
    # create_d2v_sets()
    create_bow_set()

