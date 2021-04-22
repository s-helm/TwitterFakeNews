from collections import Counter

import smart_open
from sklearn import decomposition
import multiprocessing

import gensim
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.doc2vec import TaggedDocument
import os

from NLP.NLPUtils import NLPUtils
from NLP.TextPreprocessor import TextPreprocessor

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Doc2Vec:
    def __init__(self, X, y, model_size=100, dm=1, epochs=10):
        self.X = X
        self.y = y

        self.model_size = model_size
        self.dm = dm
        self.epochs = epochs
        if X is not None and y is not None:
            self.corpus = self.read_corpus(X, y)

    def read_corpus(self, documents, y):
        """
        reads a documents and tags it with an individual label which is of the form '#_label'
        :param documents: list with tokenized documents (X)
        :param y: list with labels (y)
        :return: 
        """
        docs = []
        for i, line in enumerate(documents):
            line_new = [t.lower() for t in NLPUtils.str_list_to_list(line)
                        if "" != TextPreprocessor.remove_urls(t)
                        and "" != TextPreprocessor.remove_user_mentions(t)]
            docs.append(TaggedDocument(line_new, [str(y[i]) + "_" + str(i)]))

        return docs


    def build_model(self):
        """
        builds the Doc2Vec model
        :return: 
        """

        # min_count: ignore all words with total frequency lower than this.
        # window: the maximum distance between the current and predicted word within a sentence.
        #         Word2Vec uses a skip-gram model, and this is simply the window size of the skip-gram model.
        # size: dimensionality of the feature vectors in output. 100 is a good number.
        #       If you’re extreme, you can go up to around 400.
        # sample: threshold for configuring which higher-frequency words are randomly downsampled
        # workers: use this many worker threads to train the model

        # as suggested by authors
        window = 10
        cores = multiprocessing.cpu_count()
        self.model = gensim.models.Doc2Vec(size=self.model_size, window=window, dm=self.dm, min_count=2, workers=cores, sample=1e-4)  # use fixed learning rate
        # self.model = gensim.models.Doc2Vec(min_count=1, size=self.model_size, window=window, dm=self.dm, dm_mean=0, sample=1e-5, negative=5, workers=cores)
        # self.model = gensim.models.doc2vec.Doc2Vec(size=self.model_size, min_count=2, iter=10)
        self.model.build_vocab(self.corpus)
        # self.model.train(self.corpus, total_examples=self.model.corpus_count, epochs=self.model.iter)
        # training epochs
        for epoch in range(self.epochs):
            # self.corpus = list(self.corpus)
            # shuffle(self.corpus)
            self.model.train(documents=self.corpus, total_examples=self.model.corpus_count, epochs=self.model.iter)
            # self.model.alpha -= 0.002  # decrease the learning rate
            # self.model.min_alpha = self.model.alpha  # fix the learning rate, no deca
        # self.plotWords()

        # word = 'bad'
        # print("Most similar to {}: {}".format(word, self.model.most_similar(word)))
        # print("Most similar to {}: {}".format('trump', self.model.most_similar('trump')))
        # print("Most similar to {}: {}".format('merkel', self.model.most_similar('merkel')))
        # print(self.model.most_similar(positive=['germany','merkel'], negative=['united kingdom']))


    def create_feature_vectors(self):
        """
        creates the feature vectors 
        :return: 
        """
        nr_train_instances = len(self.X)
        train_arrays = np.zeros((nr_train_instances, self.model_size))

        for i in range(len(self.y)):
            prefix_train = str(self.y[i]) + "_" + str(i)
            train_arrays[i] = self.model.docvecs[prefix_train]
            # print(len(train_arrays[i]))

        return pd.DataFrame(data=train_arrays, columns=["tweet__d2v_{}".format(i) for i in range(self.model_size)])

    def save_model(self, model_size, dm, epochs):
        self.model.save('../data/d2v_models_testset/doc2vec_model_{}_{}_{}.d2v'.format(model_size, dm, epochs))

    def load_model(self, filename):
        self.model = gensim.models.Doc2Vec.load(filename)

    def plotWords(self):

        words_np = []
        # a list of labels (words)
        words_label = []
        for word, value in self.model.wv.vocab.items():
            words_np.append(self.model[word])
            words_label.append(word)
        print('Added %s words. Shape %s' % (len(words_np), np.shape(words_np)))

        pca = decomposition.PCA(n_components=2)
        pca.fit(words_np)
        reduced = pca.transform(words_np)

        # plt.plot(pca.explained_variance_ratio_)
        for index, vec in enumerate(reduced):
            # print ('%s %s'%(words_label[index],vec))
            if index < 100:
                x, y = vec[0], vec[1]
                plt.scatter(x, y)
                plt.annotate(words_label[index], xy=(x, y))
        plt.show()


    def perform_sanity_check(self):
        # sanity check - checks whether model is behaving in a usefully consistent manner
        ranks = []
        second_ranks = []
        for doc_id in range(len(self.corpus)):
            inferred_vector =self.model.infer_vector(self.corpus[doc_id].words)
            sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
            rank = [docid for docid, sim in sims].index(str(doc_id))
            ranks.append(rank)

            second_ranks.append(sims[1])

        print(Counter(ranks))

if __name__ == "__main__":
    # ask similarity questions to model
    d2v = Doc2Vec(X=None, y=None)
    d2v.load_model('../data/d2v_models/doc2vec_model_300_1_20.d2v')
    print(d2v.model.most_similar('merkel'))