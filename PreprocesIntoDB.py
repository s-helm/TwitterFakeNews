import json
import sys
from collections import Counter
from nltk import TweetTokenizer, sent_tokenize, re
from Database.DatabaseHandler import DatabaseHandler as db
from NLP.Emoji import Emojis
from NLP.NLPUtils import NLPUtils
from NLP.SentimentAnalysis import SentimentAnalysis
from NLP.TextPreprocessor import TextPreprocessor
from Utility.Util import chunk_list

off_abbrevs = NLPUtils.get_official_abbreviations()
sl_abbrevs = NLPUtils.get_slang_abbreviations()
curr_thread_nr = Counter()


def insert_tokenized_tweets(testset, bin=None):
    """tokenizes tweets and inserts them into the db"""

    tweets = sorted(db.get_tweets_without_feature("tokenized_text", new_only=True, testset=testset), key=lambda t: t['key_id'])

    if bin is not None:
        chunks = chunk_list(tweets, 3)
        tweets = chunks[int(bin)]
        print("Tokenize tweets from {} to {} ({} Tweets)".format(tweets[0]['key_id'], tweets[len(tweets)-1]['key_id'], len(tweets)))

    tt = TweetTokenizer()
    emojis = Emojis.read_ascii_emojis()

    count = 0
    insert_collection = list()
    for tweet in tweets:
        tokenized = TextPreprocessor.tokenize_tweet(tt, tweet['text'])
        db.insert_tokenized_text(tweet['key_id'], tokenized)
        emoji_ctr = Counter()
        for token in tokenized:
            if token in emojis:
                emoji_ctr[token] += 1
        count += 1
        insert_collection.append((json.dumps(emoji_ctr), tweet['key_id']))
        if count % 10000 == 0:
            db.insert_tweet_feature(None, 'ascii_emojis', insert_collection, 'VARCHAR(100)')
            insert_collection.clear()
            print("Tweet " + str(count) + "/" + str(len(tweets)) + " tokenized.")

    db.insert_tweet_feature(None, 'ascii_emojis', insert_collection, 'VARCHAR(100)')
    print("Tweet " + str(count) + "/" + str(len(tweets)) + " tokenized.")


def insert_sent_tokenized_tweets(testset, bin):
    """sentence tokenizes a tweets text"""

    tweets = sorted(db.get_tweets_without_feature("sent_tokenized_text", select_feature=["text","unicode_emojis", "ascii_emojis"], new_only=True, testset=testset), key=lambda t: t['key_id'])

    if bin is not None:
        chunks = chunk_list(tweets, 3)
        tweets = chunks[int(bin)]
        print(
            "Preprocess tweets from {} to {} ({} Tweets)".format(tweets[0]['key_id'], tweets[len(tweets) - 1]['key_id'],
                                                                 len(tweets)))

    count = 0
    insert_collection = list()
    for tweet in tweets:
        text = TextPreprocessor.preprocess_for_sent_tokenize(tweet['text'], tweet['unicode_emojis'], tweet['ascii_emojis'])
        tokenized = str(sent_tokenize(text))
        insert_collection.append((tokenized, tweet['key_id']))
        count += 1
        if count % 10000 == 0:
            db.insert_sent_tokenized_text(insert_collection)
            insert_collection.clear()
            print("Tweet " + str(count) + "/" + str(len(tweets)) + " sentence tokenized.")
    # insert residual
    db.insert_sent_tokenized_text(insert_collection)
    print("Tweet " + str(count) + "/" + str(len(tweets)) + " sentence tokenized.")


def insert_is_trending_topic(testset):
    print("Insert is_ww_trending_topic...")
    tweets = db.get_tweets_without_features("is_ww_trending_topic", ['additional_preprocessed_text', 'created_at'], new_only=True, testset=testset)
    db.connect_to_trending_topics_db()
    trends = db.get_worldwide_trends_by_date()

    insert_collection = list()
    count = 0
    for tweet in tweets:
        tokens = " ".join(NLPUtils.str_list_to_list(tweet['additional_preprocessed_text']))
        date = tweet['created_at'].date()

        is_trending = False
        if date in trends:
            for trend in trends[date]:
                if trend.lower() in tokens:
                    is_trending = True

        insert_collection.append((is_trending, tweet['key_id']))
        count +=1
        if count % 10000 == 0:
            db.insert_tweet_feature(None, "is_ww_trending_topic", insert_collection, 'BOOLEAN')
            insert_collection.clear()
            print("Tweet " + str(count) + "/" + str(len(tweets)) + " is_trending_topic.")
    # insert residual
    db.insert_tweet_feature(None, "is_ww_trending_topic", insert_collection, 'BOOLEAN')
    print("Tweet " + str(count) + "/" + str(len(tweets)) + " is_trending_topic")
    # c = Counter([v[0] for v in insert_collection])
    # print(c)
            # break


def insert_is_local_trending_topic():
    print("Insert is_local_trending_topic...")
    with open("resources/woeid_map.csv", 'r') as csvfile:
        import csv
        reader = csv.reader(csvfile, delimiter=';')
        map = {row[0]: row[2] for row in reader}
    db.connect_to_trending_topics_db()
    trends = db.get_trends_by_date_and_woeid()

    tweets = db.get_tweets(['tweet.key_id', 'tweet.created_at', 'user.location', 'tweet.additional_preprocessed_text'], all=True)

    insert_collection = list()

    for tweet in tweets:
        key = tweet['key_id']
        loc = tweet['location']
        date = tweet['created_at']
        tokens = " ".join(NLPUtils.str_list_to_list(tweet['additional_preprocessed_text']))

        if loc in map:
            woeid = map[loc]
            is_trending = False
            if date in trends and woeid in trends[date]:
                for trend in trends[date][woeid]:
                    if trend.lower() in tokens:
                        is_trending = True
            insert_collection.append((is_trending, key))
    db.insert_tweet_feature(None, "is_local_trending_topic", insert_collection, "BOOLEAN")
    # c = Counter([v[0] for v in insert_collection])
    # print("0: {}, 1: {}".format(c[False], c[True]))
    # print("{}/{}".format(sum(c.values()), len(tweets)))


def insert_additional_preprocessed_text(testset, bin=None):
    tweets = sorted(db.get_tweets_without_feature("additional_preprocessed_text", select_feature="pos_tags", new_only=True, testset=testset), key=lambda t: t['key_id'])

    if bin is not None:
        chunks = chunk_list(tweets, 4)
        tweets = chunks[int(bin)]
        print("Preprocess tweets from {} to {} ({} Tweets)".format(tweets[0]['key_id'], tweets[len(tweets)-1]['key_id'], len(tweets)))

    insert_collection = list()
    features = ['additional_preprocessed_text', 'contains_spelling_mistake']
    types = ['MEDIUMTEXT', 'BOOLEAN']
    count = 0

    for t in tweets:
        prepro, contains_spelling_mistake = TextPreprocessor.additional_text_preprocessing_with_pos(
            json.loads(t['pos_tags']))

        if prepro is not None:
            count += 1
            insert_collection.append((str(prepro), contains_spelling_mistake, t['key_id']))

        if count % 500 == 0:
            db.insert_tweet_features(features, insert_collection, types)
            insert_collection.clear()
            print("Tweets additional preprocessed: " + str(count) + "/" + str(len(tweets)))
    # insert residual
    db.insert_tweet_features(features, insert_collection, types)
    print("Tweets additional preprocessed: {}/{} ({} Errors)".format(count, len(tweets), len(tweets) - count))


def parse_ascii_emojis_into_db(testset, bin=None):
    """
    parses the ascii emojis in the tweets' text into a database column
    :return: 
    """
    print("Load tweets for ascii emojis...")
    tweets = sorted(db.get_tweets_without_feature(null_feature='ascii_emojis', select_feature='text', new_only=True, testset=testset), key=lambda t: t['key_id'])

    if bin is not None:
        print("Insert ascii emojis (bin {})".format(bin))
        chunks = chunk_list(tweets, 3)
        tweets = chunks[int(bin)]
        print("Preprocess tweets from {} to {} ({} Tweets)".format(tweets[0]['key_id'], tweets[len(tweets)-1]['key_id'], len(tweets)))
    else:
        print("Insert ascii emojis ({})".format(len(tweets)))

    emojis = Emojis.read_ascii_emojis()
    insert_collection = list()
    print("{} Tweets to process...".format(len(tweets)))
    count = 0
    for t in tweets:
        text = TextPreprocessor.remove_urls(t['text'])
        emojis_in_text = Emojis.find_ascii_emojis(text, emojis)
        insert_collection.append((json.dumps(emojis_in_text), t['key_id']))
        count += 1
        if count % 500 == 0:
            db.insert_tweet_feature(None, 'ascii_emojis', insert_collection, 'VARCHAR(500)')
            print("{}/{} tweets processed...".format(count, len(tweets)))
            insert_collection.clear()
    db.insert_tweet_feature(None, 'ascii_emojis', insert_collection, 'VARCHAR(500)')
    print("{}/{} tweets processed...".format(count, len(tweets)))

def replace_emoji_in_ascii_emojis(testset):
    print("Fetch all ascii emojis...")
    tweets = db.get_tweets_without_feature('ascii_emojis', 'ascii_emojis', new_only=True, testset=testset)

    insert_collection = list()
    count = 0
    for t in tweets:
        new_dict = dict()
        emojis = json.loads(t['ascii_emojis'])
        for key, value in emojis.items():
            if key != "":
                new_dict[key] = value

        insert_collection.append((json.dumps(new_dict), t['key_id']))
        count += 1
        if count % 500 == 0:
            db.insert_tweet_feature(None, 'ascii_emojis', insert_collection, 'VARCHAR(500)')
            print("{}/{} tweets processed...".format(count, len(tweets)))
            insert_collection.clear()
    db.insert_tweet_feature(None, 'ascii_emojis', insert_collection, 'VARCHAR(500)')
    print("{}/{} tweets processed...".format(count, len(tweets)))


def parse_unicode_emojis_into_db(testset, bin=None):
    """parses the unicode emojis into a database column"""
    print("Load tweets for unicode emojis")
    tweets = sorted(db.get_tweets_without_feature(null_feature='unicode_emojis', select_feature='text', new_only=True, testset=testset), key=lambda t: t['key_id'])

    if bin is not None:
        print("Insert unicode emojis (bin {})".format(bin))
        chunks = chunk_list(tweets, 3)
        tweets = chunks[int(bin)]
        print("Preprocess tweets from {} to {} ({} Tweets)".format(tweets[0]['key_id'], tweets[len(tweets)-1]['key_id'], len(tweets)))
    else:
        print("Insert unicode emojis ({})".format(len(tweets)))

    insert_collection = list()
    print("{} Tweets to process...".format(len(tweets)))
    count = 0
    for t in tweets:
        emojis_in_text = Emojis.find_unicode_emojis(t['text'])
        insert_collection.append((str(emojis_in_text), t['key_id']))
        count += 1
        if count % 500 == 0:
            db.insert_tweet_feature(None, 'unicode_emojis', insert_collection, 'VARCHAR(1000)')
            print("{}/{} tweets processed...".format(count, len(tweets)))
            insert_collection.clear()
    db.insert_tweet_feature(None, 'unicode_emojis', insert_collection, 'VARCHAR(1000)')
    print("{}/{} tweets processed...".format(count, len(tweets)))


def insert_additional_preprocessed_text_wo_stopwords(testset, bin=None):
    """
    removes stopwords from additional_preprocessed_text and inserts result into database
    :return: 
    """
    tweets = sorted(db.get_tweets_without_feature("additional_preprocessed_wo_stopwords", select_feature="additional_preprocessed_text", new_only=True, testset=testset),
           key=lambda t: t['key_id'])
    if bin is not None:
        print("Insert additional preprocessed text without stopwords (bin {})".format(bin))
        chunks = chunk_list(tweets, 3)
        tweets = chunks[int(bin)]
        print("Preprocess tweets from {} to {} ({} Tweets)".format(tweets[0]['key_id'], tweets[len(tweets)-1]['key_id'], len(tweets)))
    else:
        print("Insert additional preprocessed text without stopwords")

    insert_collection = list()
    count = 0
    for tweet in tweets:
        count += 1
        tokens = NLPUtils.str_list_to_list(tweet['additional_preprocessed_text'])
        new_tokens = list()
        for token in tokens:
            if token not in NLPUtils.get_stopwords():
                new_tokens.append(token)
        insert_collection.append((str(new_tokens), tweet['key_id']))
        if count % 10000 == 0:
            db.insert_tweet_feature(None, 'additional_preprocessed_wo_stopwords', insert_collection, None)
            print("{}/{} tweets processed...".format(count, len(tweets)))
            insert_collection.clear()
    db.insert_tweet_feature(None, 'additional_preprocessed_wo_stopwords', insert_collection, None)
    print("{}/{} tweets processed...".format(count, len(tweets)))


def insert_contains_spelling_mistake(testset, bin=None):
    """
    determines whether a tweet contains a spelling mistake or not
    :return: 
    """
    tweets = sorted(db.get_tweets_without_feature("contains_spelling_mistake", select_feature="pos_tags", new_only=True, testset=testset), key=lambda t: t['key_id'])

    if bin is not None:
        chunks = chunk_list(tweets, 3)
        tweets = chunks[int(bin)]
        print("Preprocess tweets from {} to {} ({} Tweets)".format(tweets[0]['key_id'], tweets[len(tweets)-1]['key_id'], len(tweets)))

    insert_collection = list()
    count = 0
    for t in tweets:
        count += 1
        contains_spelling_mistake = False
        pos_tags = json.loads(t['pos_tags'])
        for tags in pos_tags:
            token = tags['token']
            tag = tags['tag'].lower()
            if token != "" and not re.match('\B#\w*[a-zA-Z]+\w*', token) and token not in TextPreprocessor.PUNCTUATION and tag != ",":
                before = token
                token = TextPreprocessor.SPELL_CHECKER.correct(token, tag)
                if token != before:
                    contains_spelling_mistake = True
        insert_collection.append((contains_spelling_mistake, t['key_id']))
        if count % 10000 == 0:
            db.insert_tweet_feature(None, 'contains_spelling_mistake', insert_collection, "BOOLEAN")
            print("{}/{} tweets processed...".format(count, len(tweets)))
            insert_collection.clear()
    db.insert_tweet_feature(None, 'contains_spelling_mistake', insert_collection, "BOOLEAN")
    print("{}/{} tweets processed...".format(count, len(tweets)))


if __name__ == "__main__":
    # run from command line to use various threads
    bin = None
    if sys.argv[1:]:
        bin = sys.argv[1]

    testset=True

    # Shows which of the preprocessing columns have not yet been completed
    # db.preprocessing_values_missing(testset)
    # db.clear_column(tablename="tweet", columnname="tokenized_text")

    # parse_ascii_emojis_into_db(testset, bin)
    # parse_unicode_emojis_into_db(testset, bin)
    #
    # insert_tokenized_tweets(testset, bin)
    # insert_sent_tokenized_tweets(testset, bin)

    # based on POS-tags:
    # insert_contains_spelling_mistake(testset, bin)
    # insert_additional_preprocessed_text(testset, bin)
    # insert_additional_preprocessed_text_wo_stopwords(testset, bin)
    #
    # SentimentAnalysis.insert_sentiment_scores(testset)
    # SentimentAnalysis.insert_polarity_score(testset)
    SentimentAnalysis.insert_nr_pos_neg_words(testset=False)
    #
    # insert_is_trending_topic(testset)
    # insert_is_local_trending_topic()

    # replace_emoji_in_ascii_emojis()
