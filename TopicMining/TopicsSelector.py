import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Learning.LearningMain import perform_x_val
from Learning.LearningUtils import get_base_learners
from NLP.NLPUtils import NLPUtils
from TextRepresentation.TextModel import TextModel
from Utility.CSVUtils import load_data_from_CSV, save_df_as_csv
from Utility.TimeUtils import TimeUtils
import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from Utility.Util import get_root_directory


class TopicsSelector:
    topic_vector_dict = "../data/topic_models/"

    def __init__(self, results_file="topics_results.csv", data=None):
        time = TimeUtils.get_time()
        self.data_name = "data_for_topics_"+str(time.day)+"_"+str(time.month)+".csv"
        self.results_file = results_file
        if data is not None:
            # write_data_to_CSV(data, self.data_name)
            save_df_as_csv(data, self.data_name)

    def create_topics(self, num_topics, save=False):
        """
        performs LDA with the given number of topics and creates a dataset with only topics from it
        :param num_topics: 
        :return: 
        """
        data = load_data_from_CSV(self.data_name)
        tweet_model = TextModel()
        tweet_model.init_corpus(
            [[token for token in NLPUtils.str_list_to_list(tweet) if token != "USERMENTION" and token != "URL"] for
             tweet in data['tweet__additional_preprocessed_wo_stopwords'].tolist()])

        # LDA topics
        print("perform LDA with {} topics...".format(num_topics))
        tweet_model.perform_lda(num_topics=num_topics)
        df = tweet_model.get_topics_as_df()

        if save:
            tweet_model.save_dict_topics(num_topics)
            tweet_model.save_tf_idf_topic_model(num_topics)
            tweet_model.save_lda_model(num_topics)

        save_df_as_csv(df, get_root_directory() + "/data/data_topics_{}.csv".format(num_topics))

        data = pd.concat([data, df], axis=1)
        data = data.drop('tweet__additional_preprocessed_wo_stopwords', 1)
        return data

    @staticmethod
    def get_features(data):
        """
        selects the topic features only 
        :param data: 
        :return: 
        """
        data_columns = list(data.columns.values)

        # sel = ['user__id','tweet__fake']

        return [col for col in data_columns if re.match("tweet__topic_.*", col)]

    def optimize_num_topics(self, clfs, grid):
        """
        performs a grid search for the given numbers of topics
        :param clf: classifier to test
        :param grid: list with number of topics to test
        :return: 
        """
        # self.clear_results_csv()
        for num_topics in grid:
            data = self.create_topics(num_topics, save=True)

            for clf in clfs:
                model_name = type(clf).__name__
                print(model_name + ": ")
                f1, conf_matr = perform_x_val(data, clf, features=self.get_features(data), standardize=False)
                self.append_to_results_csv(
                    (num_topics, model_name, f1, conf_matr[0, 0], conf_matr[0, 1], conf_matr[1, 0], conf_matr[1, 1]))

    def evaluate_hdp_model(self, clfs, to_evaluate=None):
        if to_evaluate is None:
            data = load_data_from_CSV(self.data_name)
            tweet_model = TextModel()
            tweet_model.init_corpus(
                [[token for token in NLPUtils.str_list_to_list(tweet) if token != "USERMENTION" and token != "URL"] for
                 tweet in data['tweet__additional_preprocessed_wo_stopwords'].tolist()])

            # HDP topics
            tweet_model.perform_hdp()
            df = tweet_model.get_hdp_topics_as_df()

            save_df_as_csv(df, get_root_directory() + "/data/data_hdp.csv")

            data = pd.concat([data, df], axis=1)
            data = data.drop('tweet__additional_preprocessed_wo_stopwords', 1)
        else:
            data = to_evaluate


        for clf in clfs:
            model_name = "{}_hdp".format(type(clf).__name__)
            print(model_name + ": ")
            f1, conf_matr = perform_x_val(data, clf, features=self.get_features(data), standardize=False)
            self.append_to_results_csv(
                (len(data.columns)-2, model_name, f1, conf_matr[0, 0], conf_matr[0, 1], conf_matr[1, 0], conf_matr[1, 1]))


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

    def read_results_csv(self):
        """
        reads the results csv
        :param filename: 
        :return: dict with num_topic, f1, auc as keys
        """
        f = open(self.results_file, 'rt')
        try:
            reader = csv.reader(f)
            num_topics = list()
            model = list()
            f1 = list()
            conf_matrs = list()

            for row in reader:
                conf_matr = np.zeros((2, 2))
                num_topics.append(row[0])
                model.append(row[1])
                f1.append(row[2])
                conf_matr[0,0] = row[3]
                conf_matr[0,1] = row[4]
                conf_matr[1,0] = row[5]
                conf_matr[1,1] = row[6]
                conf_matrs.append(conf_matr)

            res = dict()
            res['num_topics'] = num_topics
            res['model'] = model
            res['f1'] = f1
            res['conf_matr'] = conf_matrs
            return res
        finally:
            f.close()

    def find_max_f1(self):
        csv = self.read_results_csv()

        tmp = dict()
        for i in range(len(csv['f1'])):
            if csv['model'][i] in tmp:
                tmp[csv['model'][i]].append({'model':csv['model'][i],'conf_matr':csv['conf_matr'][i], 'f1':csv['f1'][i], 'num_topics':csv['num_topics'][i]})
            else:
                tmp[csv['model'][i]] = [{'model':csv['model'][i],'conf_matr':csv['conf_matr'][i], 'f1':csv['f1'][i], 'num_topics':csv['num_topics'][i]}]

        for key, value in tmp.items():
            i = max(range(len(value)), key=lambda index: value[index]['f1'])
            print("{}: {}".format(key, value[i]))
            # print("{} & {} & {} & {} & {}".format(value[i]['model'], value[i]['num_topics'], calc_precision(value[i]['conf_matr']), calc_recall(value[i]['conf_matr']), value[i]['f1']))


    def plot_f1(self):
        """
        plots the number of topics against the f1 measure
        :return: 
        """
        csv = self.read_results_csv()


        plt.title('Number of topic in LDA vs. F1-measure')

        f1_by_model = dict()
        num_topics_by_model = dict()
        index_topics = 0
        index_model = 1
        index_f1 = 2
        for i in range(len(csv['num_topics'])):
            if csv['model'][i] in f1_by_model:
                f1_by_model[csv['model'][i]].append(csv['f1'][i])
                num_topics_by_model[csv['model'][i]].append(csv['num_topics'][i])
            else:
                f1_by_model[csv['model'][i]] = [csv['f1'][i]]
                num_topics_by_model[csv['model'][i]] = [csv['num_topics'][i]]

        count = 320
        for model, values in f1_by_model.items():
            # add x and y labels
            # plt.title('Number of Topics vs. F1-measure ('+model+')')

            if not re.match(".*_hdp", model):

                count += 1
                plt.subplot(count)
                plt.title(model)
                plt.xlabel('Number of Topics')
                plt.ylabel('F1-measure')
                plt.scatter(num_topics_by_model[model], f1_by_model[model])
                # if model != 'XGBClassifier':
                plt.scatter(num_topics_by_model["{}_hdp".format(model)], f1_by_model["{}_hdp".format(model)], color='red')

                # directory = "topics_f1"
                #
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                # plt.savefig(directory + "/"+ self.results_file[:-4] + "_"+ model +"_plot.pdf")
                # plt.close()

        plt.tight_layout(w_pad=0.5, h_pad=-1)

        plt.show()
        plt.close('all')

    def create_lda_topics_data_set(self, num_topics):
        """
        creates a temporary data set with the LDA vectors for a given number of topics
        :param num_topics: 
        :return: 
        """
        data = self.create_topics(num_topics, save=True)
        data = data[self.get_features(data)]
        save_df_as_csv(data, "{}data_set_{}_topics.csv".format(self.topic_vector_dict, num_topics))

    def create_hdp_topics_data_set(self):
        """
        creates a temporary data set with the HDP vectors
        :return: 
        """
        data = load_data_from_CSV(self.data_name)
        tweet_model = TextModel()
        tweet_model.init_corpus(
            [[token for token in NLPUtils.str_list_to_list(tweet) if token != "USERMENTION" and token != "URL"] for
             tweet in data['tweet__additional_preprocessed_wo_stopwords'].tolist()])

        # HDP topics
        tweet_model.perform_hdp()
        data = tweet_model.get_hdp_topics_as_df()
        data = data[self.get_features(data)]
        save_df_as_csv(data, self.topic_vector_dict+"data_set_hdp.csv")
        tweet_model.save_hdp_model()
        tweet_model.save_dict_topics('hdp')
        tweet_model.save_tf_idf_topic_model('hdp')

if __name__ == "__main__":
    # create and evaluate topics
    cols = ['tweet__additional_preprocessed_wo_stopwords', 'user__id', 'tweet__fake']
    data = load_data_from_CSV("../data/data_set_tweet_user_features.csv")
    data = data[cols]
    data = data.reset_index(drop=True)
    #
    num_topics_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    #
    clfs = get_base_learners()

    t = TopicsSelector(results_file="results_topics_final.csv", data=data)
    t.optimize_num_topics(clfs=clfs, grid=num_topics_list)
    # t.evaluate_hdp_model(clfs=clfs)

    # plot results
    # t = TopicsSelector(results_file="results/results_topics_final.csv")
    # t.plot_f1()

    # t = TopicsSelector(data=data)
    # t.create_lda_topics_data_set(90)


    # evaluate a topic dataset
    # t = TopicsSelector(results_file="test.csv", data=None)
    # topic_vectors = load_data_from_CSV("../data/topics_data/data_topics_90.csv")
    #
    # data = join_label_and_group(data=topic_vectors)
    # perform_x_val(data, get_base_learners('nb'), t.get_features(data))


