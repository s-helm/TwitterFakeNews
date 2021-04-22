import copy
import re
import gensim
import pandas as pd
import time
import matplotlib.pyplot as plt
from gensim import models, similarities, corpora
from gensim.models import VocabTransform, Word2Vec
from NLP.NLPUtils import NLPUtils
from TextRepresentation.Util import Util
from Utility.CSVUtils import load_data_from_CSV
from Utility.Util import get_root_directory


class TextModel:

    folder_d2v_models = '../data/d2v_models/'
    folder_topic_models = '../data/topic_models/'
    folder_bow_models = '../data/bow_models/'
    folder_tfidf = '../data/tfidf_models/'

    def __init__(self, corpus=None, dictionary=None):
        self.dictionary = dictionary
        self.bow_filtered_dict = None

        # corpora
        self.corpus = corpus
        self.corpus_tfidf = None
        self.corpus_lda = None
        self.corpus_hdp = None
        self.corpus_lsi = None

        # models
        self.lda = None
        self.lsi = None
        self.hdp = None
        self.tf_idf = None
        self.vt = None
        self.word2vec = None
        self.doc2vec = None

        self.num_topics = None

    def init_corpus(self, tweets):
        dictionary, corpus = Util.create_corpus(tweets)
        self.corpus = corpus
        self.dictionary = dictionary

        # if save:
        #     # self.save_corpus()
        #     self.save_dict()

    def calc_tf_idf(self):
        """
        calculates TF-IDF
        :return: 
        """

        self.tf_idf = models.TfidfModel(self.corpus)
        corpus_tfidf = self.tf_idf[self.corpus]
        self.corpus_tfidf = corpus_tfidf

        # if save_model:
        #     self.save_tf_idf_model(stopwords=stopwords)
        #
        # if save_corpus:
        #     self.save_corpus_tfidf()
        return corpus_tfidf

    def perform_lsi(self, num_topics=10):
        """
        performs Latent Semantic Indexing
        :param corpus_tfidf: 
        :param dictionary: 
        :param num_topics: 200-500 is recommended
        :return: 
        """
        if self.corpus_tfidf is None:
           self.calc_tf_idf()
        self.lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary,
                                   num_topics=num_topics)  # initialize an LSI transformation
        self.corpus_lsi = self.lsi[
            self.corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
        self.lsi.print_topics(num_topics)
        # for doc in corpus_lsi:  # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
        # print(doc)
        return self.corpus_lsi

    def perform_lda(self, num_topics=2):
        """
        performs Latent Dirichlet Allocation
        :param num_topics: 
        :return: lda
        """
        self.num_topics = num_topics
        if self.corpus_tfidf is None:
           self.calc_tf_idf()
        #
        # self.lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary,
        #                            num_topics=num_topics)

        self.lda = models.LdaMulticore(self.corpus_tfidf, id2word=self.dictionary,
                                   num_topics=num_topics, passes=2, minimum_probability=0.01)
        self.corpus_lda = self.lda[self.corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lda
        self.lda.print_topics(num_topics)
        # for doc in self.corpus_lda:  # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly (topic, degree towards topic)
        #     print(doc)
        return self.corpus_lda


    def perform_hdp(self):
        """
        performs Hierarchical Dirichlet Process
        :param num_topics: 
        :return: lda
        """
        if self.corpus_tfidf is None:
           self.calc_tf_idf()

        self.hdp = models.HdpModel(self.corpus_tfidf, id2word=self.dictionary)
        self.corpus_hdp = self.hdp[self.corpus_tfidf]
        # self.hdp.print_topics(num_topics)
        # for doc in corpus_lda:  # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly (topic, degree towards topic)
        #     print(doc)
        return self.corpus_hdp

    def perform_word2vec(self, tweets):
        self.word2vec = Word2Vec(tweets, size=100)
        return self.word2vec

    def get_most_similar_doc(self, doc):
        """
        calculates the similarity of the doc to the docs in the corpus
        :param doc: 
        :return: the most similar one according to LSI
        """
        self.calc_tf_idf()
        lsi = self.perform_lsi(2)

        vec_bow = self.dictionary.doc2bow(doc.lower().split())
        vec_lsi = lsi[vec_bow]  # convert the query to LSI space
        print(vec_lsi)

        index = similarities.MatrixSimilarity(lsi[self.corpus])  # transform corpus to LSI space and index it

        sims = index[vec_lsi]  # perform a similarity query against the corpus
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        print(sims)  # print sorted (document number, similarity score) 2-tuples
        return sims[0]

    def get_tf_idf_series(self):
        """
        calculates tf-idf scores, sums them up and normalizes by tweet length
        :return: 
        """
        if self.corpus_tfidf is None:
            self.calc_tf_idf()
        tf_idf_sums = list()
        for tweet in self.corpus_tfidf:
            if len(tweet) == 0:
                tf_idf_sums.append(0)
            else:
                tf_idf_sums.append(sum(token[1] for token in tweet) / len(tweet))
        return pd.Series(tf_idf_sums).values

    def get_topics_as_df(self):
        """
        creates a list of series out of the LDA corpus
        :return: list of pandas Series. Each series is one topic 
        """

        numpy_matrix = gensim.matutils.corpus2dense(self.corpus_lda, num_terms=self.num_topics)

        cols = ["tweet__topic_{}".format(i) for i in range(len(numpy_matrix))]
        numpy_matrix = numpy_matrix.transpose()
        print(numpy_matrix.shape)
        return pd.DataFrame(numpy_matrix, columns=cols)

    def get_hdp_topics_as_df(self):
        """
        creates a list of series out of the HDP corpus
        :return: list of pandas Series. Each series is one topic 
        """

        num_topics = len(self.hdp.print_topics(-1))
        print("Number of topics HDP: {}".format(num_topics))
        numpy_matrix = gensim.matutils.corpus2dense(self.corpus_hdp, num_terms=num_topics)

        cols = ["tweet__topic_hdp_{}".format(i) for i in range(len(numpy_matrix))]
        numpy_matrix = numpy_matrix.transpose()
        print(numpy_matrix.shape)
        return pd.DataFrame(numpy_matrix, columns=cols)

    def build_bag_of_words(self, variant, tf_idf=True, min_doc_frequency=100, no_above=0.5, keep_n=10000):
        """
        creates a bag of words representation out of the corpus based on tf-idf. Filters based on the given filter
        :param min_doc_frequency: filter words that are in less than min_doc_frequency docs
        :param no_above: filter words that are more than no_above percent of the documents
        :param keep_n: number of most frequent words to keep after filtering
        :return: 
        """

        if self.corpus_tfidf is None and tf_idf:
            self.calc_tf_idf()
            self.save_tf_idf_bow_model(variant)

        self.save_bow_dict(variant)

        new_dict = copy.deepcopy(self.dictionary)
        new_dict.filter_extremes(no_below=min_doc_frequency, no_above=no_above, keep_n=keep_n)

        old2new = {self.dictionary.token2id[token]: new_id for new_id, token in new_dict.items()}
        self.vt = VocabTransform(old2new)
        self.bow_filtered_dict = new_dict
        if tf_idf:
            corpus = self.vt[self.corpus_tfidf]
        else:
            corpus = self.vt[self.corpus]

        self.save_vt_model(variant)
        self.save_bow_filtered_dict(variant)

        # for doc in corpus:
        #     print([dictionary.id2token[token[0]] for token in doc])
        return self.build_bag_of_words_df(corpus)

    def build_bag_of_words_df(self, corpus):
        """
        builds the dataframe for the bag of words
        :param corpus: 
        :param dictionary: 
        :param term_vectors: 
        :return: 
        """
        term_vectors = dict()

        for word in self.bow_filtered_dict:
            term_vector = list()
            for docs in corpus:
                found = None
                for token in docs:
                    if token[0] == word:
                        found = token[1]
                if found is None:
                    # term_vector.append(None)
                    term_vector.append(0)
                else:
                    term_vector.append(found)
            term_vectors[self.bow_filtered_dict.id2token[word]] = term_vector

        # return {key: pd.Series(word, index=index).to_sparse() for key, word in term_vectors.items()}
        return {key: pd.Series(word) for key, word in term_vectors.items()}

    def optimize_nr_of_topics_lda(self, hold_out_set, total_docs, nr_of_topics_grid=None, topics_lower_bound=None, topics_upper_bound=None):
        """
        Evaluates each number of topics between topics_lower_bound and topics_upper_bound based on the log perplexity
        
        :param nr_of_topics_grid: a list with specified nr of topics values
        :param topics_lower_bound: min number of topics to test
        :param topics_upper_bound: max number of topics to test
        :param hold_out_set: hold out data set to evaluate
        :param total_docs: total nr of docs cropus + hold out
        :return: 
        """
        hold_out_set = [self.dictionary.doc2bow(tweet) for tweet in hold_out_set]

        grid_x = list()
        grid_y = list()
        if nr_of_topics_grid:
            coll = nr_of_topics_grid
        else:
            coll = range(topics_lower_bound, topics_upper_bound+1)

        print("Start tying...")
        for i in coll:
            start_time = time.time()
            self.perform_lda(num_topics=i)
            log_perplexity = self.lda.log_perplexity(chunk=hold_out_set, total_docs=total_docs)
            elapsed_time = time.time() - start_time
            print("Nr of topics: {} - Perplexity {} ({})".format(i, log_perplexity, elapsed_time))
            doc_perplexity = self.lda.bound(hold_out_set)
            grid_x.append(i)
            grid_y.append(log_perplexity)
        self.plot_perplexity(grid_x, grid_y)


    def plot_perplexity(self, x, y):
        # add title
        plt.title('Relationship Between Number of Topics and Perplexity')

        # add x and y labels
        plt.xlabel('Number of topics')
        plt.ylabel('Perplexity')

        plt.scatter(x, y)
        plt.show()


    def save_lda_model(self, num_topics, prefix=''):
        self.lda.save(TextModel.folder_topic_models+prefix+'lda_{}_topics.model'.format(num_topics))

    def load_lda_model(self, num_topics, prefix=''):
        self.lda = gensim.models.LdaModel.load(TextModel.folder_topic_models+prefix+'lda_{}_topics.model'.format(num_topics))


    def save_hdp_model(self):
        self.hdp.save(TextModel.folder_topic_models+'hdp.model')

    def load_hdp_model(self):
        self.hdp = gensim.models.LdaModel.load(TextModel.folder_topic_models + 'hdp.model')


    def save_tf_idf_model(self):
        self.tf_idf.save(self.folder_tfidf+'tf_idf.model')

    def load_tf_idf_model(self):
        self.tf_idf = gensim.models.TfidfModel.load(self.folder_tfidf+'tf_idf.model')


    def save_tf_idf_topic_model(self, num_topics, prefix = ''):
        self.tf_idf.save(self.folder_topic_models+prefix+'tf_idf_{}.model'.format(num_topics))

    def load_tf_idf_topic_model(self, num_topics, prefix = ''):
        self.tf_idf = gensim.models.TfidfModel.load(self.folder_topic_models+prefix+'tf_idf_{}.model'.format(num_topics))


    def save_tf_idf_bow_model(self, variant):
        self.tf_idf.save(TextModel.folder_bow_models+'tf_idf_bow_{}.model'.format(variant))

    def load_tf_idf_bow_model(self, variant):
        self.tf_idf = gensim.models.TfidfModel.load(TextModel.folder_bow_models+'tf_idf_bow_{}.model'.format(variant))


    def save_vt_model(self, variant):
        self.vt.save(TextModel.folder_bow_models+'vt_{}.model'.format(variant))

    def load_vt_model(self, variant):
        self.vt = gensim.models.VocabTransform.load(TextModel.folder_bow_models+'vt_{}.model'.format(variant))


    def load_bigram_dict(self):
        self.dictionary = gensim.corpora.Dictionary.load(TextModel.folder_bow_models + 'bigram.dict')

    def save_bigram_dict(self):
        self.dictionary.save(TextModel.folder_bow_models + 'bigram.dict')


    def save_tf_idf_bigram_model(self):
        self.tf_idf.save(TextModel.folder_bow_models+'tf_idf_bigram.model')

    def load_tf_idf_bigram_model(self):
        self.tf_idf = gensim.models.TfidfModel.load(TextModel.folder_bow_models+'tf_idf_bigram.model')


    def save_dict(self):
        self.dictionary.save(self.folder_tfidf+'word.dict')

    def load_dict(self):
        self.dictionary = gensim.corpora.Dictionary.load(self.folder_tfidf+'word.dict')


    def save_dict_topics(self, num_topics, prefix=''):
        self.dictionary.save(self.folder_topic_models + prefix + 'words_{}.dict'.format(num_topics))

    def load_dict_topics(self, num_topics, prefix=''):
        self.dictionary = gensim.corpora.Dictionary.load(self.folder_topic_models + prefix + 'words_{}.dict'.format(num_topics))


    def save_bow_dict(self, variant):
        self.dictionary.save(TextModel.folder_bow_models+'bow_{}.dict'.format(variant))

    def load_bow_dict(self, variant):
        self.dictionary = gensim.corpora.Dictionary.load(TextModel.folder_bow_models+'bow_{}.dict'.format(variant))


    def save_bow_filtered_dict(self, variant):
        self.bow_filtered_dict.save(TextModel.folder_bow_models+'bow_filtered_{}.dict'.format(variant))

    def load_bow_filtered_dict(self, variant):
        self.bow_filtered_dict = gensim.corpora.Dictionary.load(TextModel.folder_bow_models+'bow_filtered_{}.dict'.format(variant))


    def save_corpus(self):
        corpora.MmCorpus.serialize(TextModel.folder_topic_models + 'corpus.mm', self.corpus)

    def load_corpus(self):
        self.corpus = corpora.MmCorpus(TextModel.folder_topic_models + 'corpus.mm')


    def save_corpus_tfidf(self):
        corpora.MmCorpus.serialize(TextModel.folder_topic_models + 'corpus_tfidf.mm', self.corpus_tfidf)

    def load_corpus_tfidf(self):
        self.corpus_tfidf = corpora.MmCorpus(TextModel.folder_topic_models + 'corpus_tfidf.mm')


def find_most_similar_real_to_fake():
    """
    finds the most similar real news tweet to a tweet from the fake news class
    :return: 
    """
    data = load_data_from_CSV('../data/data_set_tweet_user_features.csv')
    data = data.reset_index(drop=True)
    real = list(data.loc[data['tweet__fake'] == 0].index)
    fake = list(data.loc[data['tweet__fake'] == 1].index)

    tm = TextModel()
    tm.load_dict()
    tm.load_corpus()
    tm.calc_tf_idf()

    for fake_index in fake:
        tfidf1 = tm.corpus_tfidf[fake_index]
        max = 0
        best_index = None

        for i in real:
            index = similarities.MatrixSimilarity([tfidf1], num_features=len(tm.dictionary))
            tfidf2 = tm.corpus_tfidf[i]
            sim = index[tfidf2]

            if sim > max:
                max = sim
                best_index = i
        print("{} Best {}: {}".format(fake_index, best_index, max))


def find_most_similar_real_to_fake_in_holdout():
    train = load_data_from_CSV("../data/data_set_tweet_user_features.csv")[['tweet__additional_preprocessed_text', 'tweet__id', 'tweet__fake']]
    train = train.loc[train['tweet__fake'] == 0]
    train.reset_index(drop=True)
    train_tmp = train['tweet__additional_preprocessed_text'].tolist()

    tweets = [[token for token in NLPUtils.str_list_to_list(tweet) if
                    token != "USERMENTION" and token != "URL" and not re.match('^\d+$', token)] for tweet in train_tmp]

    tm = TextModel()
    tm.init_corpus(tweets)
    tm.calc_tf_idf()

    test = load_data_from_CSV(get_root_directory() + "/data/testset_tweet_user_features.csv")
    test = test.loc[test['tweet__fake'] == 1]
    test = test.sample(frac=1).reset_index(drop=True)
    tmp = test['tweet__additional_preprocessed_text'].tolist()
    fake_test = [[token for token in NLPUtils.str_list_to_list(tweet) if
                    token != "USERMENTION" and token != "URL" and not re.match('^\d+$', token)] for tweet in tmp]

    rand_sel = fake_test[:99]
    print(test.index)
    print(train.index)


    doc_bow = [tm.dictionary.doc2bow(tweet) for tweet in rand_sel]
    corpus_fake_testset = tm.tf_idf[doc_bow]

    ids = list()
    for j in range(len(corpus_fake_testset)):
        max = -1
        best = None
        for i in range(len(tm.corpus_tfidf)):
            sim = gensim.matutils.cossim(corpus_fake_testset[j], tm.corpus_tfidf[i])
            if sim > max:
                max = sim
                best = i
        real_id = train.iloc[[best]]['tweet__id'].values[0]
        ids.append(real_id)
        fake_id = test.iloc[[j]]['tweet__id'].values[0]
        print("Best for {}: {} ({})".format(fake_id, real_id, max))

    with open("ids.txt", "w") as output:
        output.write(str(ids))


if __name__ == "__main__":
    find_most_similar_real_to_fake_in_holdout()
