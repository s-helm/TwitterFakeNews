import csv
import re
import numpy as np
import pandas as pd

from Learning.EvaluationMetrics import calc_recall, calc_precision
from Learning.LearningMain import perform_x_val
from Learning.LearningUtils import get_base_learners
from NLP.NLPUtils import NLPUtils
from TextRepresentation.Doc2Vec import Doc2Vec
from TextRepresentation.TextModel import TextModel
from Utility.CSVUtils import save_df_as_csv, load_data_from_CSV
from Utility.TimeUtils import TimeUtils

class TextModelSelector:

    text_model_vector_dir = "../data/text_data/"

    def __init__(self, results_file, data=None):
        time = TimeUtils.get_time()
        self.data_name = "data_for_text_model_"+str(time.day)+"_"+str(time.month)+".csv"
        self.results_file = results_file
        if data is not None:
            # write_data_to_CSV(data, self.data_name)
            save_df_as_csv(data, self.data_name)

    def create_bag_of_words(self, min_doc_frequency, no_above):
        """
        creates a bag of words with different filters
        :param num_topics: 
        :return: 
        """
        data = load_data_from_CSV(self.data_name)
        tweet_model = TextModel()

        tweets = [NLPUtils.str_list_to_list(tweet) for tweet in data['tweet__additional_preprocessed_wo_stopwords'].tolist()]
        for i in range(len(tweets)):
            tweets[i] = [token for token in tweets[i] if token != 'URL' and token != 'USERMENTION' and not re.match('^\d+$',token)]

        tweet_model.init_corpus(tweets)

        print("build bag of words... [min_doc_frequency={},no_above={}]".format(min_doc_frequency, no_above))
        term_vectors = tweet_model.build_bag_of_words(variant='unigram', min_doc_frequency=min_doc_frequency, no_above=no_above, keep_n=500)

        # num_cols = len(term_vectors.items())
        for key, series in term_vectors.items():
            data['tweet__contains_{}'.format(key)] = series

        # for col in data.columns:
        #     print(data[col].describe())

        data = data.drop('tweet__additional_preprocessed_wo_stopwords', 1)
        return data


    def create_bag_of_words_bigram(self, min_doc_frequency, no_above):
        """
        creates a bigram bag of words with different filters
        :param num_topics:
        :return:
        """
        data = load_data_from_CSV(self.data_name)
        bigram_model = TextModel()
        tmp = data['tweet__additional_preprocessed_text'].tolist()
        tweets = list()
        for tweet in tmp:
            tmp_tweet = NLPUtils.str_list_to_list(tweet)
            tmp_tweet = ['NUMBER' if re.match('^\d+$', token) else token for token in tmp_tweet]
            tweets.append(tmp_tweet)

        bigram_tweets = [NLPUtils.generate_n_grams(tweet, 2) for tweet in tweets]
        bigram_model.init_corpus(bigram_tweets)

        print("build bag of words... [min_doc_frequency={},no_above={}]".format(min_doc_frequency, no_above))
        term_vectors = bigram_model.build_bag_of_words(variant='bigram', min_doc_frequency=min_doc_frequency, no_above=no_above, keep_n=300)
        for key, vector in term_vectors.items():
            data['tweet__contains_bigram_{}'.format(re.sub(" ", "_", str(key)))] = vector

        data = data.drop('tweet__additional_preprocessed_text', 1)
        return data


    def create_doc2vec(self, model_size, dm, epochs):
        """
        creates Doc2Vec with given parameters
        :param model_size: 
        :param dm: 
        :param epochs: 
        :return: Doc2Vec dataset
        """
        data = load_data_from_CSV(self.data_name)
        # print(data.columns)
        # print(data.head())
        X = data['tweet__additional_preprocessed_text'].tolist()
        y = data['tweet__fake'].tolist()
        print("Create Doc2Vec (size: {}, dm: {}, epochs: {})...".format(model_size, dm, epochs))
        doc2Vec = Doc2Vec(X, y, model_size, dm, epochs)
        doc2Vec.build_model()
        doc2Vec.save_model(model_size=model_size, dm=dm, epochs=epochs)
        fv = doc2Vec.create_feature_vectors()
        print(data.index)
        print(fv.index)

        save_df_as_csv(fv, '../data/d2v_models_{}_{}_{}.csv'.format(model_size, dm, epochs))

        data = data.reset_index(drop=True)
        fv.reset_index(drop=True)
        print(data.shape)
        print(fv.shape)
        data = pd.concat([data, fv], axis=1)

        # for col in data.columns:
        #     print(data[col].describe())
        return data

    @staticmethod
    def get_features(data, doc2vec=False):
        """
        selects the topic features only 
        :param data: 
        :return: 
        """
        data_columns = list(data.columns.values)

        # sel = ['user__id','tweet__fake']
        if doc2vec:
            return [col for col in data_columns if 'tweet__d2v_' in col]
        else:
            return [col for col in data_columns if re.match("tweet__contains_.*", col)]

    def optimize_text_model(self, clfs, grid, variant=0):
        """
        performs a grid search for the given numbers of topics
        :param clf: classifier to test
        :param grid: list with number of topics to test
        :param variant: 0: bag of words, 1: bigrams, 2: doc2vec
        :return: 
        """
        # self.clear_results_csv()

        # doc2vec
        if variant == 2:
            for size in grid['size']:
                for epochs in grid['epochs']:
                    for dm in grid['dm']:
                        data = self.create_doc2vec(size, dm, epochs)
                        # data = load_data_from_CSV('../data/d2v_models_{}_{}_{}.csv'.format(size, dm, epochs))
                        # data = join_label_and_group(data)
                        for clf in clfs:
                            f1, conf_matr = perform_x_val(data, clf, features=self.get_features(data, doc2vec=True))
                            model_name = type(clf).__name__
                            print("Model {}".format(model_name))
                            print("Model size: {}, Epochs: {}, Architecture: {}".format(size, epochs, dm))
                            print("F1: {}".format(f1))
                            # print("AUC: {}".format(auc))
                            config = "Doc2Vec_{}_{}_{}_{}".format(size, epochs, dm, model_name)
                            self.append_to_results_csv(
                                (config, f1, conf_matr[0, 0], conf_matr[0, 1], conf_matr[1, 0], conf_matr[1, 1]))

        # bag of words
        else:
            for min_doc_freq in grid['min_doc_frequency']:
                for no_above in grid['no_above']:
                    if variant == 0:
                        data = self.create_bag_of_words(min_doc_frequency=min_doc_freq, no_above=no_above)
                        # for col in data.columns:
                        #     print(data[col].describe())
                    elif variant == 1:
                        data = self.create_bag_of_words_bigram(min_doc_frequency=min_doc_freq, no_above=no_above)
                    for clf in clfs:
                        model_name = type(clf).__name__
                        try:
                            f1, conf_matr = perform_x_val(data, clf, features=self.get_features(data))
                            print("Model {}".format(model_name))
                            print("min_doc_frequency: {}".format(min_doc_freq))
                            print("no_above: {}".format(no_above))
                            print("F1: {}".format(f1))
                            # print("AUC: {}".format(auc))
                            conf = "{}_{}_{}".format(min_doc_freq, no_above, model_name)
                            self.append_to_results_csv(
                                (conf, f1, conf_matr[0, 0], conf_matr[0, 1], conf_matr[1, 0], conf_matr[1, 1]))
                        except Exception as e:
                            print("Variant: {}({}):{}".format(variant, model_name, e))

    def append_to_results_csv(self, row):
        """
        appends at the end of the results csv file
        :param row: 
        :param filename: 
        :return: 
        """
        f = open(self.results_file, 'a')
        try:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(row)
        finally:
            f.close()

    def clear_results_csv(self):
        """
        clears a csv file
        :param filename: 
        :return: 
        """
        with open(self.results_file, "w") as my_empty_csv:
            print("{} cleared.".format(self.results_file))

    def read_results_csv(self, variant):
        """
        reads the results csv
        :param filename: 
        :return: dict with num_topic, f1, auc as keys
        """
        f = open(self.results_file, 'rt')
        try:
            reader = csv.reader(f)
            results = list()
            for row in reader:
                if variant == 0:
                    tmp = row[0].split("_")
                    min_doc_freq = tmp[0]
                    no_above = tmp[1]
                    algo = tmp[2]
                    # num_cols = tmp[3]

                    f1 = row[1]
                    conf_matr = np.zeros((2, 2))
                    conf_matr[0,0] = row[2]
                    conf_matr[0,1] = row[3]
                    conf_matr[1,0] = row[4]
                    conf_matr[1,1] = row[5]

                    res = dict()
                    res['algo'] = algo
                    res['min_doc_freq'] = int(min_doc_freq)
                    res['no_above'] = float(no_above)
                    # res['num_cols'] = num_cols
                    res['f1'] = f1
                    res['prec'] = calc_precision(conf_matr)
                    res['rec'] = calc_recall(conf_matr)
                    res['conf_matr'] = conf_matr
                    results.append(res)
                if variant == 2:
                    tmp = row[0].split("_")
                    algo = tmp[4]
                    size = tmp[1]
                    epochs = tmp[2]
                    dm = tmp[3]

                    f1 = row[1]
                    conf_matr = np.zeros((2, 2))
                    conf_matr[0, 0] = row[2]
                    conf_matr[0, 1] = row[3]
                    conf_matr[1, 0] = row[4]
                    conf_matr[1, 1] = row[5]

                    res = dict()
                    res['algo'] = algo
                    res['size'] = size
                    res['dm'] = dm
                    res['epochs'] = epochs
                    res['f1'] = f1
                    res['prec'] = calc_precision(conf_matr)
                    res['rec'] = calc_recall(conf_matr)
                    res['conf_matr'] = conf_matr
                    results.append(res)
            return sorted(results, key=lambda k: (k['algo'], k['f1']))
            # return sorted(results, key=lambda k: (k['algo'], k['min_doc_freq'], k['no_above']))

        finally:
            f.close()

    def find_max_f1(self, algo, variant):
        csv = self.read_results_csv(variant=variant)
        csv = sorted(csv, key=lambda k: (k['algo'], k['f1']), reverse=True)

        for res in csv:
            if res['algo'] == type(get_base_learners(algo)).__name__:
                return res

    def create_doc2vec_data_set(self, size, dm, epochs, save=True):
        """
        creates temporary doc2vec data set
        :param size: 
        :param dm: 
        :param epochs: 
        :param save: if True, saves dataset as csv
        :return: 
        """
        data = self.create_doc2vec(model_size=size, dm=dm, epochs=epochs)
        if save:
            data = data[self.get_features(data, doc2vec=True)]
            save_df_as_csv(data, "{}data_set_{}_{}_{}.csv".format(self.text_model_vector_dir, size, dm, epochs))


    @staticmethod
    def create_d2v_datasets(save=True):
        """
        creates the Doc2Vec datasets with best parameters 
        :param save: if True, stores the datasets otherwise only stores Doc2Vec models
        :return: 
        """
        data = load_data_from_CSV("../data/data_set_tweet_user_features.csv")
        cols_d2v = ['tweet__additional_preprocessed_text', 'tweet__fake']
        data = data[cols_d2v]
        t = TextModelSelector(data=data, results_file=None)
        t.create_doc2vec_data_set(size=300, dm=0, epochs=20, save=save)
        t.create_doc2vec_data_set(size=100, dm=0, epochs=20, save=save)
        t.create_doc2vec_data_set(size=200, dm=0, epochs=20, save=save)


if __name__ == "__main__":
    clfs = get_base_learners()
    data = load_data_from_CSV("../data/data_set_tweet_user_features.csv")
    data = data.reset_index(drop=True)

    # ---------bag-of-words-(variant-0)-----------------------------------------------------------
    # cols_bow = ['tweet__additional_preprocessed_wo_stopwords', 'user__id', 'tweet__fake']
    # data = data[cols_bow]
    #
    # grid = {"min_doc_frequency":[500], "no_above":[0.4]}
    # # grid = {"min_doc_frequency":[500,2000,3500], "no_above":[0.4,0.5]}
    # t_bow = TextModelSelector(results_file='results_bow_final.csv')
    # t_bow.optimize_text_model(clfs=clfs, grid=grid, variant=0)

    #----------bigrams-(variant-1)--------------------------------------------------------------
    # cols_bow_bi = ['tweet__additional_preprocessed_text', 'user__id', 'tweet__fake']
    # data = data[cols_bow_bi]
    #
    # grid = {"min_doc_frequency":[250], "no_above":[0.4]}
    # t_bow_bi = TextModelSelector(results_file="results_bow_bi_final.csv", data=data)
    # t_bow_bi.optimize_text_model(clfs=clfs, grid=grid, variant=1)

    #---------doc2vec-(variant-2)----------------------------------------------------------------
    cols_d2v = ['tweet__additional_preprocessed_text', 'user__id', 'tweet__fake']
    data = data[cols_d2v]

    grid = {"size":[100, 200, 300], "dm":[0,1], "epochs":[20]}
    t = TextModelSelector(results_file="results_doc2vec_final.csv", data=data)
    t.optimize_text_model(clfs=clfs, grid=grid, variant=2)


    # TextModelSelector.create_d2v_datasets(save=False)