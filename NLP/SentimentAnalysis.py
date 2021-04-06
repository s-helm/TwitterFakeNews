import json
import re

from nltk import TabTokenizer
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk
from textblob import TextBlob

from Database.DatabaseHandler import DatabaseHandler
from NLP.Emoji import Emojis


class SentimentAnalysis:

    EMOJI_SENTS = Emojis.read_unicode_emoji_sents_map()
    EMOJI_SENTS_ASCII = Emojis.read_ascii_emoji_sents_map()

    @staticmethod
    def score_word_sentiment(word, pos_tag, tweet_pos):
        """
        :param word: word to score
        :param pos_tag: n - NOUN, v - VERB, a - ADJECTIVE, r - ADVERB
        :param tweet_pos: POS tagged tokens
        :return: sentiment score (pos_score - neg_score)
        """
        from NLP.TextPreprocessor import TextPreprocessor
        allowed_tags = ['a','n','v','r']
        pos_tag = pos_tag.lower()
        if pos_tag in allowed_tags:
            word = TextPreprocessor.lemmatize(word, pos_tag)
            synset = SentimentAnalysis.disambiguate_word(word, pos_tag, tweet_pos)
            if synset is None:
                return 0.0
            sent_word = swn.senti_synset(synset.name())
            score = sent_word.pos_score()-sent_word.neg_score()
            # print(sent_word)
            # print("pos: {}".format(sent_word.pos_score()))
            # print("neg: {}".format(sent_word.neg_score()))
            # print("obj: {}".format(sent_word.obj_score()))
            # print("score: {}".format(score))
            return score
        elif re.match("U\+.{4,5}", word):
            # scores the unicode emojis
            if pos_tag == 'e':
                return SentimentAnalysis.EMOJI_SENTS[word]['score']
        elif word in SentimentAnalysis.EMOJI_SENTS_ASCII:
            return SentimentAnalysis.EMOJI_SENTS_ASCII[word]['score']

        return 0.0

    @staticmethod
    def disambiguate_word(word, tag, tweet_pos):
        """
        disambiguates a word in a tweet
        :param word: word to disambiguate
        :param tag: pos tag of the word
        :param tweet_pos: list with POS tagged tokens
        :return: best matching Synset
        """
        sent = [key["token"] for key in tweet_pos]
        return lesk(sent, word, tag)

    @staticmethod
    def score_tweet_sentiment(tweet_pos):
        """
        Scores a tweet according to its sentiment
        :param tweet_pos: POS tagged tweet
        :return: summed up sentiment score, number of sentiment words
        """
        score = 0
        nr_sent_words = 0
        for t in tweet_pos:
            score_t = SentimentAnalysis.score_word_sentiment(t["token"], t["tag"],tweet_pos)
            # print("{} (score: {})".format(t["token"], score_t))
            score += score_t
            if score_t != 0:
                nr_sent_words += 1
        if nr_sent_words != 0:
            return SentimentAnalysis.normalize_score(score/nr_sent_words), nr_sent_words
        else:
            return 0, 0

    @staticmethod
    def count_pos_neg_sentiment_words(tweet_pos):
        """
        counts positive and negative sentiment words
        :param tweet_pos: POS tagged tweet
        :return: summed up sentiment score, number of sentiment words
        """
        nr_pos_words = 0
        nr_neg_words = 0
        for t in tweet_pos:
            score_t = SentimentAnalysis.score_word_sentiment(t["token"], t["tag"],tweet_pos)
            # print("{} (score: {})".format(t["token"], score_t))
            if score_t > 0:
                nr_pos_words += 1
            if score_t < 0:
                nr_neg_words += 1
        return nr_pos_words, nr_neg_words


    @staticmethod
    def insert_nr_pos_neg_words(testset):
        print('Insert pos neg words')
        tweets = DatabaseHandler.get_tweets_without_feature(null_feature="nr_pos_sentiment_words", select_feature="pos_tags", new_only=True, testset=testset)

        insert_collection = list()

        feature_names = ['nr_pos_sentiment_words','nr_neg_sentiment_words']
        types = ['INT','INT']

        count = 0
        length = len(tweets)
        for t in tweets:
            count += 1

            id = t["key_id"]
            try:
                text = json.loads(t["pos_tags"])

                pos_count, neg_count = SentimentAnalysis.count_pos_neg_sentiment_words(text)
                insert_collection.append((pos_count, neg_count, id))
            except:
                pass

            if count % 5000 == 0:
                DatabaseHandler.insert_tweet_features(feature_names=feature_names, values=insert_collection,
                                                      sql_types=types)
                insert_collection.clear()
                print("{}/{} tweets processed.".format(count, length))
        # insert residual
        DatabaseHandler.insert_tweet_features(feature_names=feature_names, values=insert_collection, sql_types=types)
        print("{}/{} tweets processed.".format(count, length))


    @staticmethod
    def insert_sentiment_scores(testset):
        """
        inserts a column with the sentiment score as well as the number of sentiment words in the text
        :return: -
        """
        print("Insert sentiment score")
        tweets = DatabaseHandler.get_tweets_without_feature(null_feature="sentiment_score", select_feature="pos_tags", new_only=True, testset=testset)

        insert_collection = list()

        feature_names = ["sentiment_score", "nr_of_sentiment_words"]
        types = ["FLOAT", "INT"]

        count = 0
        length = len(tweets)
        for t in tweets:
            count += 1

            id = t["key_id"]
            text = json.loads(t["pos_tags"])
            score, nr_sent_words = SentimentAnalysis.score_tweet_sentiment(text)
            insert_collection.append((score, nr_sent_words, id))

            if count % 500 == 0:
                DatabaseHandler.insert_tweet_features(feature_names=feature_names, values=insert_collection, sql_types=types)
                insert_collection.clear()
                print("{}/{} tweets processed.".format(count, length))
        # insert residual
        DatabaseHandler.insert_tweet_features(feature_names=feature_names, values=insert_collection, sql_types=types)
        print("{}/{} tweets processed.".format(count, length))

    @staticmethod
    def assess_subjectivity(pos_tags):
        """
        determines the polarity of sentence with the TextBlob library
        :param pos_tags: 
        :return: 
        """
        from textblob.en.sentiments import PatternAnalyzer
        from textblob.en.sentiments import NaiveBayesAnalyzer
        from NLP.NLPUtils import NLPUtils
        words = list()
        for token in pos_tags:
            word = token['token']
            pos_tag = token['tag']
            allowed_tags = ['a','n','v','r']
            if pos_tag.lower() in allowed_tags:
                # word = TextPreprocessor.lemmatize(word, pos_tag.lower())
                pass
            Emojis.remove_unicode_emojis(word)
            # if pos_tag != '#' and pos_tag != '@' and pos_tag != 'U' and pos_tag != 'E' and word not in NLPUtils.get_punctuation():
            if pos_tag != '#' and pos_tag != '@' and pos_tag != 'U' and word not in NLPUtils.get_punctuation():
                words.append(word)
        text = "\t".join(words)
        # print(text)
        tokenizer = TabTokenizer()
        testimonial3 = TextBlob(text, analyzer=PatternAnalyzer(), tokenizer=tokenizer)
        # print(testimonial3.sentiment)
        polarity = SentimentAnalysis.normalize_score(testimonial3.sentiment.polarity)
        subjectivity = testimonial3.sentiment.subjectivity
        return polarity, subjectivity

    @staticmethod
    def insert_polarity_score(testset):
        """insert polarity"""
        print("Insert subjectivity score")
        tweets = DatabaseHandler.get_tweets_without_feature(null_feature="subjectivity_score", select_feature="pos_tags", new_only=True, testset=testset)

        insert_collection = list()
        count = 0
        length = len(tweets)
        for t in tweets:
            count += 1

            id = t["key_id"]
            text = json.loads(t["pos_tags"])
            polarity, subjectivity = SentimentAnalysis.assess_subjectivity(text)
            insert_collection.append((subjectivity, id))

            if count % 500 == 0:
                DatabaseHandler.insert_tweet_feature(None, feature_name="subjectivity_score", value=insert_collection,
                                                      sql_type="FLOAT")
                insert_collection.clear()
                print("{}/{} tweets processed.".format(count, length))
        # insert residual
        DatabaseHandler.insert_tweet_feature(None, feature_name="subjectivity_score", value=insert_collection,
                                             sql_type="FLOAT")
        print("{}/{} tweets processed.".format(count, length))

    @staticmethod
    def normalize_score(score):
        """
        normalizes the score to a range from 0 to 1
        :param score: 
        :return: 
        """
        return (score + 1)/2

