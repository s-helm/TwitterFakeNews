from collections import defaultdict, Counter
from math import log
from operator import itemgetter
from NLP.NLPUtils import NLPUtils


class TermWeighter:

    @staticmethod
    def user_idfs(tweets_by_user, normalize=False):
        """calculates a single idf for each individual user"""
        user_idfs = dict()
        for user, tweets in tweets_by_user.items():
            user_idfs[user] = TermWeighter.calc_idf(tweets, normalize=normalize)
        return user_idfs

    @staticmethod
    def tweet_tf_idf_sum(tweet, idf, normalize=True):
        """calculates the tf idf score of a tweet by summing up the individual scores
        -tweet: tweet tokenized
        -idf: dict with terms and inverse document frequency
        -normalize: whether length normalization should be performed or not"""
        if len(tweet) == 0:
            return 0
        return sum([s[1] for s in TermWeighter.calc_tf_idf_scores(tweet, idf, normalize=normalize)])/len(tweet)

    @staticmethod
    def calc_idf(corpus, normalize=False):
        """calculates inverse document frequency (idf) for each term
        -corpus: list of lists
        -normalize: flag to perform normalization. If a word occurs in every doc, then log(1) = 0 = idf"""
        df = defaultdict(int)
        for doc in corpus:
            terms = set(doc)
            # document frequency of terms
            for term in terms:
                df[term] += 1
        idf = {}
        # inverse document frequency
        for term, term_df in df.items():
            if normalize:
                # (uses +1 normalization)
                idf[term] = 1 + log(len(corpus) / term_df)
            else:
                # no normalization
                idf[term] = log(len(corpus) / term_df)
        return idf

    @staticmethod
    def calc_tf_idf_scores(doc, idf, normalize=True):
        """calculates tf-idf
        -doc: list of tokens
        -idf: idf score
        -normalize: flag to perform document length normalization (converts the TF to the probability P(t|d)"""
        tf = Counter(doc)
        if normalize:
            tf = {term: tf_value/len(doc) for term, tf_value in tf.items()}
        try:
            tfidf = {term: tf_value*idf[term] for term, tf_value in tf.items()}
        except KeyError as e:
            print(e)
            print(doc)
            print(idf)
        # sort by second item in the tuple, descending order
        return sorted(tfidf.items(), key=itemgetter(1), reverse = True)

    @staticmethod
    def get_top_100_tf_idf_score(doc, idf, normalize=False):
        """
        returns a list with the top 100 TF-IDF scores
        :param doc: 
        :param idf: 
        :param normalize: 
        :return: 
        """
        return TermWeighter.calc_tf_idf_scores(doc, idf, normalize)[:100]


    @staticmethod
    def term_frequencies(tweet_tokens):
        """returns the top 20 most frequent terms. Stopwords are removed."""
        tokens_list = list()

        tf = Counter()
        for t in tweet_tokens:
            punct = NLPUtils.get_punctuation()
            stopwords = NLPUtils.get_stopwords()
            if t not in punct and t not in stopwords:
                tf.update(t)
                tokens_list.extend(t)

        for tag, count in tf.most_common(20):
            print("{}: {}".format(tag, count))

