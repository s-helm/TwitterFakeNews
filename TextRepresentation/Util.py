import json
import os

import gensim
from gensim import corpora, similarities


class Util:

    @staticmethod
    def create_corpus(tweets):
        """
        stores a corpus
        :param corpus: 
        :return: dictionary, corpus
        """
        dictionary = corpora.Dictionary(tweets)
        corpus = [dictionary.doc2bow(tweet) for tweet in tweets]
        return dictionary, corpus

    @staticmethod
    def save_index(index):
        index.save('corpus.index')
        print("Index saved")

    @staticmethod
    def load_index():
        return similarities.MatrixSimilarity.load('corpus.index')

    @staticmethod
    def corpus_to_numpy(corpus, number_of_corpus_features):
        return gensim.matutils.corpus2dense(corpus, num_terms=number_of_corpus_features)

    @staticmethod
    def numpy_to_corpus(numpy_matrix):
        return gensim.matutils.Dense2Corpus(numpy_matrix)

    @staticmethod
    def find_key_by_value(dict, search_value):
        """
        returns key for a given value
        :param dict: dictionary to search for key
        :param search_value: value to search for
        :return: key, or None if key not in dictionary
        """
        for key, value in dict.items():
            if value == search_value:
                return key
        return None

    @staticmethod
    def write_topic_to_json(topic, topic_nr, total_topics):
        directory = "plots/topics_{}".format(total_topics)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + "/topic_nr_{}.json".format(topic_nr), 'w') as fp:
            json.dump(topic, fp)
