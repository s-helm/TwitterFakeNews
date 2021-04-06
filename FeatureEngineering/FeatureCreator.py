import json
from collections import Counter
from functools import lru_cache

from statistics import median

import pandas as pd
import re

from nltk import ngrams

from Database.DatabaseHandler import DatabaseHandler as db
from NLP.Emoji import Emojis
from NLP.NLPUtils import NLPUtils
from NLP.TermWeighter import TermWeighter

from NLP.TextParser import TextParser
from NLP.TextPreprocessor import TextPreprocessor
from TextRepresentation.TextModel import TextModel
from Utility.JsonUtils import json_string_to_counter
from Utility.TimeUtils import TimeUtils
from Utility.UrlUtils import UrlUtils

punctuation = NLPUtils.get_punctuation()


def create_features(data, conf=None):
    print("...create features...")

    data = data.reset_index(drop=True)
    # TWEET features
    print("tweet features...")

    # if there is one sentence this could either be a single sentence or no real sentence in the text
    data['tweet__nr_of_sentences'] = data['tweet__sent_tokenized_text'].map(lambda x: len(NLPUtils.str_list_to_list(x)))

    # data = data.drop('tweet__sent_tokenized_text', 1)

    # emojis
    print("tweet features - emojis...")
    print("nr of unicode emojis")
    data['tweet__nr_of_unicode_emojis'] = data['tweet__unicode_emojis'].map(lambda x: len(NLPUtils.str_list_to_list(x)))
    print("contains unicode emojis")
    data['tweet__contains_unicode_emojis'] = data['tweet__nr_of_unicode_emojis'].map(lambda x: x > 0)
    print("face positive emojis")
    data['tweet__contains_face_positive_emojis'] = data['tweet__unicode_emojis'].map(
        lambda x: Emojis.unicode_emoji_in_category(NLPUtils.str_list_to_list(x), 'face-positive') > 0)
    print("face negative emojis")
    data['tweet__contains_face_negative_emojis'] = data['tweet__unicode_emojis'].map(
        lambda x: Emojis.unicode_emoji_in_category(NLPUtils.str_list_to_list(x), 'face-negative') > 0)
    print("face neutral emojis")
    data['tweet__contains_face_neutral_emojis'] = data['tweet__unicode_emojis'].map(
        lambda x: Emojis.unicode_emoji_in_category(NLPUtils.str_list_to_list(x), 'face-neutral') > 0)

    # print("drop column tweet__unicode_emojis")
    # data = data.drop('tweet__unicode_emojis', 1)

    print("nr of ascii emojis")
    data['tweet__nr_of_ascii_emojis'] = data['tweet__ascii_emojis'].map(
        lambda x: sum(json_string_to_counter(x).values()))
    print("contains ascii emojis")
    data['tweet__contains_ascii_emojis'] = data['tweet__nr_of_ascii_emojis'].map(lambda x: x > 0)

    # print("drop column tweet__ascii_emojis")
    # data = data.drop('tweet__ascii_emojis', 1)

    data['tweet__tokenized_um_url_removed'] = data['tweet__tokenized_text'].map(
        lambda x: remove_um_url(NLPUtils.str_list_to_list(x)))

    data['tweet__has_place'] = ~(pd.isnull(data['tweet__place_id']))
    data['tweet__is_reply_to_status'] = ~(pd.isnull(data['tweet__in_reply_to_status_id']))
    data['tweet__is_quoted_status'] = ~(pd.isnull(data['tweet__quoted_status_id']))
    data['tweet__is_retweeted_status'] = ~(pd.isnull(data['tweet__retweeted_status_id']))
    data['tweet__has_location'] = ~(pd.isnull(data['tweet__location_id']))
    data['tweet__possibly_sensitive_news'] = data['tweet__possibly_sensitive'].map(lambda x: get_possibly_sensitive(x))
    data['tweet__no_text'] = data['tweet__tokenized_um_url_removed'].map(
        lambda x: len(NLPUtils.str_list_to_list(x)) == 0)

    data['tweet__nr_of_words'] = data['tweet__tokenized_um_url_removed'].map(
        lambda x: get_nr_of_words(NLPUtils.str_list_to_list(x)))

    # pos tag related features
    print("POS features")
    data['tweet__nr_tokens'] = data['tweet__pos_tags'].map(
        lambda x: get_nr_of_words(NLPUtils.str_list_to_list(remove_um_url([t['token'] for t in json.loads(x)]))))
    data['tweet__ratio_adjectives'] = data.apply(
        lambda x: get_tag_ratio(x['tweet__pos_tags'], 'A', x['tweet__nr_tokens']),
        axis=1)
    data['tweet__ratio_nouns'] = data.apply(lambda x: get_tag_ratio(x['tweet__pos_tags'], 'N', x['tweet__nr_tokens']),
                                            axis=1)
    data['tweet__ratio_verbs'] = data.apply(lambda x: get_tag_ratio(x['tweet__pos_tags'], 'V', x['tweet__nr_tokens']),
                                            axis=1)
    data['tweet__contains_named_entities'] = data['tweet__pos_tags'].map(
        lambda x: tweet_contains_named_entities(json.loads(x)))
    data['tweet__contains_pronouns'] = data['tweet__pos_tags'].map(lambda x: "U" in [t['tag'] for t in json.loads(x)])

    # POS trigrams
    print("tweet POS trigrams")
    trigram_vectors = find_frequent_pos_trigrams(data, min_doc_frequency=1000, no_above=0.4, keep_n=100)
    for key, vector in trigram_vectors.items():
        data['tweet__contains_pos_trigram_{}'.format(re.sub(" ", "_", str(key)))] = vector

    # text/word length
    print("tweet text/word length...")
    data['tweet__avg_word_length'] = data['tweet__tokenized_um_url_removed'].map(
        lambda x: tweet_avg_word_length(NLPUtils.str_list_to_list(x)))
    data['tweet__nr_of_slang_words'] = data['tweet__tokenized_um_url_removed'].map(
        lambda x: len(slang_words_in_tweet(NLPUtils.str_list_to_list(x))))
    data['tweet__ratio_uppercase_letters'] = data['tweet__tokenized_um_url_removed'].map(
        lambda x: tweet_ratio_uppercase_letters(x))
    data['tweet__ratio_capitalized_words'] = data.apply(
        lambda x: tweet_ratio_capitalized_words(x), axis=1)
    data['tweet__ratio_all_capitalized_words'] = data.apply(
        lambda x: tweet_ratio_all_capitalized_words(x), axis=1)

    # data = data.drop('tweet__tokenized_um_url_removed', 1)

    data['tweet__nr_of_tokens'] = data['tweet__tokenized_text'].map(lambda x: len(NLPUtils.str_list_to_list(x)))
    data['tweet__ratio_tokens_before_after_prepro'] = data.apply(
        lambda x: tweet_ratio_tokens_before_after_prepro(NLPUtils.str_list_to_list(x['tweet__tokenized_text']),
                                                         NLPUtils.str_list_to_list(
                                                             x['tweet__additional_preprocessed_wo_stopwords'])), axis=1)

    print("features regarding tweet text")
    data['tweet__text_length'] = data.apply(
        lambda row: get_tweet_text_length(row['tweet__text'], row['tweet__is_reply_to_status']), axis=1)
    data['tweet__percent_of_text_used'] = data['tweet__text_length'] / 140
    data['tweet__ratio_words_tokens'] = data['tweet__nr_of_words'] / data['tweet__nr_of_tokens']

    # url
    print("tweet features - url...")
    data['tweet__nr_of_urls'] = data.apply(lambda x: tweet_nr_of_urls(x['tweet__entities_id'], x['tweet__text']),
                                           axis=1)
    data['tweet__contains_urls'] = data['tweet__nr_of_urls'].map(lambda x: x > 0)
    data['tweet__avg_url_length'] = data.apply(
        lambda x: tweet_avg_url_length(x['tweet__entities_id'], x['tweet__text']), axis=1)
    data['tweet__url_only'] = (data['tweet__nr_of_urls'] > 0) & data['tweet__no_text']

    # stock symbol
    print("tweet stock symbol...")
    data['tweet__contains_stock_symbol'] = data['tweet__text'].map(lambda x: bool(tweet_find_stock_mention(x)))

    # punctuation
    print("tweet features - punctuation...")
    data['tweet__nr_of_punctuations'] = data['tweet__text'].map(
        lambda x: sum(get_nr_of_punctuation(x).values()))
    data['tweet__contains_punctuation'] = data['tweet__nr_of_punctuations'].map(lambda x: x > 0)
    data['tweet__ratio_punctuation_tokens'] = data.apply(
        lambda x: ratio_punctuation_tokens(x['tweet__tokenized_text'], x['tweet__nr_of_tokens']), axis=1)
    data['tweet__nr_of_exclamation_marks'] = data['tweet__text'].map(
        lambda x: get_nr_of_punctuation(x)['!'])
    data['tweet__contains_exclamation_mark'] = data['tweet__nr_of_exclamation_marks'].map(lambda x: x > 0)
    data['tweet__multiple_exclamation_marks'] = data['tweet__nr_of_exclamation_marks'].map(lambda x: x > 1)
    data['tweet__nr_of_question_marks'] = data['tweet__text'].map(
        lambda x: get_nr_of_punctuation(x)['?'])
    data['tweet__contains_question_mark'] = data['tweet__nr_of_question_marks'].map(lambda x: x > 0)
    data['tweet__multiple_question_marks'] = data['tweet__nr_of_question_marks'].map(lambda x: x > 1)

    # further NLP
    print("tweet features - further NLP...")

    data['tweet__contains_character_repetitions'] = data['tweet__text'].map(
        lambda x: tweet_contains_character_repetitions(x))
    data['tweet__contains_slang'] = data['tweet__nr_of_slang_words'].map(
        lambda x: 0 < x)

    data['tweet__is_all_uppercase'] = data['tweet__text'].map(lambda x: is_upper(x))
    data['tweet__contains_uppercase_text'] = data['tweet__text'].map(lambda x: contains_all_uppercase(x))

    data['tweet__contains_number'] = data['tweet__additional_preprocessed_text'].map(
        lambda x: contains_number(NLPUtils.str_list_to_list(x)))
    data['tweet__contains_quote'] = data['tweet__text'].map(lambda x: contains_quote(x))
    # data = data.drop('tweet__text', 1)

    # media
    print("tweet features - media...")
    data['tweet__nr_of_medias'] = data['tweet__entities_id'].map(lambda x: tweet_nr_of_medias(x))
    data['tweet__contains_media'] = data['tweet__nr_of_medias'].map(lambda x: x > 0)

    # user mentions
    print("tweet features - user mentions...")
    data['tweet__nr_of_user_mentions'] = data['tweet__entities_id'].map(lambda x: tweet_nr_of_user_mentions(x))
    data['tweet__contains_user_mention'] = data['tweet__nr_of_user_mentions'].map(lambda x: x > 0)

    # hashtags
    print("tweet features - hashtag...")
    top_100_hashtags = db.get_most_popular_hashtags_across_users(100)

    data['tweet__nr_of_hashtags'] = data['tweet__entities_id'].map(lambda x: tweet_nr_of_hashtags(x))
    data['tweet__contains_hashtags'] = data['tweet__nr_of_hashtags'].map(lambda x: x > 0)
    data['tweet__nr_of_popular_hashtag'] = data['tweet__entities_id'].map(
        lambda x: tweet_nr_of_hashtags_in_popular_hashtags(x, top_100_hashtags))
    data['tweet__contains_popular_hashtag'] = data['tweet__nr_of_popular_hashtag'].map(
        lambda x: x > 0)


    data['tweet__additional_preprocessed_is_empty'] = data['tweet__additional_preprocessed_wo_stopwords'].map(
        lambda x: len(NLPUtils.str_list_to_list(x)) == 0)

    # sentiment related
    data['tweet__contains_sentiment'] = data['tweet__sentiment_score'].map(lambda x: x != 0.5)
    data['tweet__ratio_pos_sentiment_words'] = data.apply(
        lambda x: tweet_ratio_sentiment_words(x['tweet__nr_pos_sentiment_words'], x['tweet__nr_of_sentiment_words']), axis=1)
    data['tweet__ratio_neg_sentiment_words'] = data.apply(
        lambda x: tweet_ratio_sentiment_words(x['tweet__nr_neg_sentiment_words'], x['tweet__nr_of_sentiment_words']), axis=1)

    data['tweet__ratio_stopwords'] = data.apply(lambda x: tweet_ratio_tokens_before_after_prepro(
        NLPUtils.str_list_to_list(x['tweet__additional_preprocessed_text']),
        NLPUtils.str_list_to_list(x['tweet__additional_preprocessed_wo_stopwords'])), axis=1)

    # time features
    print("tweet features - time...")
    data['tweet__day_of_week'] = data['tweet__created_at'].map(
        lambda x: int(TimeUtils.mysql_to_python_datetime(x).weekday()))
    # DataHandler.store_data(data)

    data['tweet__day_of_month'] = data['tweet__created_at'].map(lambda x: TimeUtils.mysql_to_python_datetime(x).day)
    data['tweet__day_of_year'] = data['tweet__created_at'].map(
        lambda x: TimeUtils.mysql_to_python_datetime(x).timetuple().tm_yday)
    data['tweet__month_of_year'] = data['tweet__created_at'].map(lambda x: TimeUtils.mysql_to_python_datetime(x).month)
    data['tweet__year'] = data['tweet__created_at'].map(lambda x: TimeUtils.mysql_to_python_datetime(x).year)
    data['tweet__am_pm'] = data.apply(
        lambda x: TimeUtils.is_pm(TimeUtils.mysql_to_python_datetime(x['tweet__created_at']), x['user__utc_offset']),
        axis=1)
    data['tweet__hour_of_day'] = data.apply(
        lambda x: TimeUtils.hour_of_day(TimeUtils.mysql_to_python_datetime(x['tweet__created_at']),
                                        x['user__utc_offset']),
        axis=1)
    data['tweet__quarter_of_year'] = data['tweet__month_of_year'].map(lambda x: tweet_quarter(x))

    # relative
    data['tweet__created_days_ago'] = data['tweet__created_at'].map(
        lambda x: TimeUtils.days_ago(TimeUtils.mysql_to_python_datetime(x), relative_to="2017-07-24 16:01"))
    # absolute
    # data['tweet__created_days_ago'] = data['tweet__created_at'].map(
    #     lambda x: TimeUtils.days_ago(TimeUtils.mysql_to_python_datetime(x), relative_to="2017-07-24 16:01"))


    # tf-idf features
    # print("tweet tf-idf features...")
    tweet_model = TextModel()

    tweet_model.init_corpus(
        [NLPUtils.str_list_to_list(tweet) for tweet in data['tweet__additional_preprocessed_wo_stopwords'].tolist()])
    data['tweet__tf_idf_sum'] = tweet_model.get_tf_idf_series()

    tweets_by_user = dict()
    for index, row in data.iterrows():
        user = row['user__id']
        tweet = NLPUtils.str_list_to_list(row['tweet__additional_preprocessed_wo_stopwords'])
        if user in tweets_by_user:
            tweets_by_user[user].append(tweet)
        else:
            tweets_by_user[user] = [tweet]

    user_idfs = TermWeighter.user_idfs(tweets_by_user)
    data['tweet__tf_idf_sum_grouped_by_user'] = data.apply(
        lambda x: TermWeighter.tweet_tf_idf_sum(
            NLPUtils.str_list_to_list(x['tweet__additional_preprocessed_wo_stopwords']), user_idfs[x['user__id']]),
        axis=1)
    del tweets_by_user

    # # bag-of-words unigram
    # if 'uni' in conf['text_models']:
    #     del tweet_model
    #     tweet_model = TextModel()
    #     tweet_model.init_corpus([[token for token in NLPUtils.str_list_to_list(tweet) if
    #                               token != "USERMENTION" and token != "URL" and not re.match('^\d+$', token)] for tweet
    #                              in
    #                              data['tweet__additional_preprocessed_wo_stopwords'].tolist()])
    #
    #     print("tweet bag-of-words unigram")
    #     term_vectors = tweet_model.build_bag_of_words(variant='unigram', min_doc_frequency=conf['uni']['min_doc_freq'],
    #                                                   no_above=conf['uni']['no_above'], keep_n=500)
    #     for key, series in term_vectors.items():
    #         data['tweet__contains_{}'.format(key)] = series
    #
    #     print(data.shape)

    # # LDA topics
    # if 'topic_model' in conf['text_models']:
    #
    #     print("tweet LDA topics...")
    #     del tweet_model
    #     tweet_model = TextModel()
    #     tweet_model.init_corpus([[token for token in NLPUtils.str_list_to_list(tweet) if
    #                               token != "USERMENTION" and token != "URL" and not re.match('^\d+$', token)] for tweet
    #                              in
    #                              data['tweet__additional_preprocessed_wo_stopwords'].tolist()])
    #     if conf['topic_model']['lda']:
    #         tweet_model.perform_lda(num_topics=conf['topic_model']['num_topics'])
    #         df = tweet_model.get_topics_as_df()
    #         data = pd.concat([data, df], axis=1)
    #     else:
    #         tweet_model.perform_hdp()
    #         df = tweet_model.get_hdp_topics_as_df()
    #         data = pd.concat([data, df], axis=1)

    # print(data.shape)
    # if 'doc2vec' in conf['text_models']:
    #     X = data['tweet__additional_preprocessed_wo_stopwords'].tolist()
    #     y = data['tweet__fake'].tolist()
    #     doc2Vec = Doc2Vec(X, y, model_size=conf['doc2vec']['size'], dm=conf['doc2vec']['dm'],
    #                       epochs=conf['doc2vec']['epochs'])
    #     doc2Vec.build_model()
    #     fv = doc2Vec.create_feature_vectors()
    #     data = pd.concat([data, fv], axis=1)
    #
    #     data = data.drop('tweet__additional_preprocessed_wo_stopwords', 1)
    #     DataHandler.store_data(data)

    # bag-of-words bigram
    print("tweet bag-of-words bigram")
    bigram_model = TextModel()
    tmp = data['tweet__additional_preprocessed_text'].tolist()
    bigram_tweets = [NLPUtils.generate_n_grams(NLPUtils.str_list_to_list(tweet), 2) for tweet in tmp]
    bigram_model.init_corpus(bigram_tweets)

    data['tweet__bigram_tf_idf_sum'] = bigram_model.get_tf_idf_series()

    # if 'bi' in conf['text_models']:
    #     term_vectors = bigram_model.build_bag_of_words(min_doc_frequency=conf['bi']['min_doc_freq'],
    #                                                    no_above=conf['bi']['no_above'], keep_n=500)
    #     for key, vector in term_vectors.items():
    #         data['tweet__contains_bigram_{}'.format(re.sub(" ", "_", str(key)))] = vector

    # data = data.drop('tweet__additional_preprocessed_text', 1)
    # DataHandler.store_data(data)

    return data


def tweet_ratio_all_capitalized_words(x):
    nr_of_words = x['tweet__nr_of_words']
    if nr_of_words == 0:
        return 0
    else:
        return nr_of_all_capitalized_words(
            NLPUtils.str_list_to_list(x['tweet__tokenized_um_url_removed'])) / nr_of_words


def tweet_ratio_capitalized_words(x):
    nr_of_words = x['tweet__nr_of_words']

    if nr_of_words == 0:
        return 0
    else:
        return nr_of_capitalized_words(NLPUtils.str_list_to_list(x['tweet__tokenized_um_url_removed'])) / nr_of_words


def tweet_ratio_uppercase_letters(x):
    upper_count = nr_of_cased_characters(NLPUtils.str_list_to_list(x))
    if upper_count == 0:
        return 0
    else:
        return tweet_count_upper_letters(NLPUtils.str_list_to_list(x)) / upper_count


def bigram_in_tweet_pos(pos_tags, bigram):
    """counts the number of POS bigrams in a tweet and normalizes it by the length of the tweet"""
    pos_tags = [token['tag'] for token in pos_tags]
    bigrams = list(ngrams(pos_tags, 2))
    count = 0
    if bigram in bigrams:
        count += 1

    if len(bigrams) == 0:
        return None
    return count / len(bigrams)


def find_frequent_pos_trigrams(data, min_doc_frequency=10000, no_above=0.5, keep_n=100):
    """
    finds trigrams that meat the frequency threshold for POS tags
    :param data: pandas dataframe
    :param threshhold: to minimum number of accurances in the data
    :return: 
    """

    tweets_tags = data['tweet__pos_tags'].tolist()

    trigram_pos = []
    for tweet in tweets_tags:
        tweet = json.loads(tweet)
        pos_tags = [token['tag'] for token in tweet]
        trigram_pos.append(NLPUtils.generate_n_grams(pos_tags, 3))

    pos_tri_model = TextModel()
    pos_tri_model.init_corpus(trigram_pos)
    return pos_tri_model.build_bag_of_words(variant='pos_trigram',tf_idf=True, min_doc_frequency=min_doc_frequency, no_above=no_above,
                                            keep_n=keep_n)


@lru_cache(maxsize=None)
def get_user_lang_counts(user_id):
    """counts the languages used by a user. Does not count lang 'und' since language
    is automatically detected by twitter and almost every account contains a tweet with undefined language"""
    lang_counts = db.get_user_lang_counts(user_id)

    count = 0
    for c in lang_counts:
        if c['lang'] != 'und':
            count += 1
    return count


def favourites_per_follower(x):
    """return #favourties/#followers"""
    return x['user__favourites_count'] / x['user__followers_count']


def friends_per_follower(x):
    """return #favourties/#followers"""
    return x['user__friends_count'] / x['user__followers_count']


def friends_per_favourite(x):
    """return #friends/#favourties, if favourites = 0 returns 0"""
    if x['user__favourites_count'] == 0:
        return 0
    else:
        return x['user__friends_count'] / x['user__favourites_count']


def contains_hashtag(x):
    """manually detect hashtags"""
    result = TextParser.find_all_hashtags(str(x))
    if result:
        return True
    else:
        return False


def contains_user_mention(x):
    result = TextParser.find_all_user_mentions(str(x))
    if result:
        return True
    else:
        return False


def contains_url(x):
    # SearchStr = '(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    search_str = '(?P<url>https?://[^\s]+)'
    result = re.search(search_str, str(x))
    if result:
        return True
    else:
        return False


def string_length(x):
    if x is not None:
        return len(str(x))
    else:
        return 0


def is_translator_type(x):
    # potentially more classes
    if x == 'regular':
        return True
    else:
        return False


def tweet_contains_hashtags(entity_id):
    if pd.isnull(entity_id):
        return False
    else:
        hashtags = db.get_hashtags_of_tweet(entity_id)
        if len(hashtags) > 0:
            return True
        else:
            return False


def tweet_nr_of_hashtags(entity_id):
    if pd.isnull(entity_id):
        return 0
    else:
        hashtags = db.get_hashtags_of_tweet(entity_id)
        if hashtags is not None:
            return len(hashtags)
        else:
            return 0


def tweet_nr_of_urls(entity_id, text):
    if pd.isnull(entity_id):
        return 0
    else:
        length_1 = len(TextParser.find_all_urls(str(text)))
        length_2 = len(db.get_urls_of_tweet(entity_id))

        if length_1 > length_2:
            return length_1
        else:
            return length_2


def tweet_contains_urls(entity_id, text):
    if pd.isnull(entity_id):
        return False
    else:
        length_1 = len(TextParser.find_all_urls(str(text)))
        length_2 = len(db.get_urls_of_tweet(entity_id))

        return (length_1 > 0) or (length_2 > 0)


def tweet_contains_media(entity_id):
    if pd.isnull(entity_id):
        return False
    else:
        media = db.get_media(entity_id)
        if len(media) > 0:
            return True
    return False


def tweet_nr_of_medias(entity_id):
    if pd.isnull(entity_id):
        return 0
    else:
        media = db.get_media(entity_id)
        if media is not None:
            return len(media)
    return 0


def tweet_contains_user_mention(entity_id):
    if pd.isnull(entity_id):
        return False
    else:
        media = db.get_user_mentions(entity_id)
        if len(media) > 0:
            return True
    return False


def tweet_nr_of_user_mentions(entity_id):
    if pd.isnull(entity_id):
        return 0
    else:
        um = db.get_user_mentions(entity_id)
        if um is not None:
            return len(um)
    return 0


def tweet_avg_url_length(entity_id, text):
    """returns the length of an url in a tweet.
    If tweet contains more than one url, average length is returned"""
    if pd.isnull(entity_id):
        return 0
    else:
        parsed_urls = TextParser.find_all_urls(str(text))
        urls_from_tweepy = db.get_urls_of_tweet(entity_id)

        length_1 = len(parsed_urls)
        length_2 = len(urls_from_tweepy)

        if length_1 == 0 and length_2 == 0:
            return 0
        elif length_1 > length_2:
            sum = 0
            for i in parsed_urls:
                sum += len(i)
            return sum / len(parsed_urls)
        else:
            sum = 0
            for i in urls_from_tweepy:
                sum += len(i)
            return sum / len(urls_from_tweepy)


def tweet_contains_link_to_users_website(user_url, entities_id):
    """returns true if one of the urls in the tweet link to the users website"""
    tweet_urls = db.get_manual_expanded_urls_of_tweet(entities_id)
    user_urls = db.get_manual_expanded_url_by_url(user_url)

    if len(user_urls) > 0:
        user_domain = UrlUtils.extract_domain(user_urls[0])
        for i in tweet_urls:
            domain = UrlUtils.extract_domain(i)
            if domain == user_domain:
                return True
    return False


def tweet_avg_expanded_url_length(entity_id):
    """returns the avg length of the expanded url"""
    urls = db.get_manual_expanded_urls_of_tweet(entity_id)
    sum = 0
    for i in urls:
        sum += len(i)
    if len(urls) == 0:
        return 0
    return sum / len(urls)


def one_of_tweet_urls_is_expandable(entity_id):
    """returns true if a tweet contains an shortened link"""
    urls = db.get_urls(entity_id)
    for u in urls:
        url = u['url']
        expanded_url = u['expanded_url']
        manual_expanded_url = u['manual_expanded_url']

        if re.match("www.*", url):
            url = "http://" + url

        url = re.sub("www\.", '', url)
        expanded_url = re.sub("www\.", '', expanded_url)
        manual_expanded_url = re.sub("www\.", '', manual_expanded_url)

        if url == expanded_url and expanded_url == manual_expanded_url:
            return False
        elif expanded_url is None:
            return False
        elif url != expanded_url:
            return True
        elif manual_expanded_url is None:
            return False
        elif url != manual_expanded_url:
            return True
        elif expanded_url != manual_expanded_url:
            return True


def get_top_level_domain_type(tld):
    """returns the type of the top level domain
    0: generic
    1: country-code"""
    type = UrlUtils.get_top_level_domain_type(tld)

    if type == "country-code":
        return 1
    else:
        return 0


def get_tweet_text_length(text, quoted):
    t = str(text)
    t = TextPreprocessor.unescape_html(t)
    norm_text = TextParser.normalize(t)
    length = len(norm_text)

    # user mentions in quoted tweets do not count
    if quoted == 1:
        ums = TextParser.find_all_user_mentions(t)
        for um in ums:
            norm = TextParser.normalize(um)
            length -= len(norm)
    return length


@lru_cache(maxsize=None)
def get_avg_post_time(user_id):
    """returns the average post time in hours of day"""
    avg_times = db.get_post_times_of_user(user_id)
    tz = db.get_user_utc_offset(user_id)
    sum = 0
    for t in avg_times:
        local_time = TimeUtils.utc_to_timezone(t, tz)
        sum += local_time.hour

    return sum / len(avg_times)


@lru_cache(maxsize=None)
def get_tweets_per_month(user_id):
    """returns the average number of tweets per month.
    First and last month in the database are omitted since it can't be assumed that they are complete."""
    times = sorted(db.get_post_times_of_user(user_id))
    counts = dict()
    month = ""
    first_omitted = False
    for t in times:
        # dont count first
        if month != str(t.month) + "_" + str(t.year) and month != "":
            first_omitted = True
        month = str(t.month) + "_" + str(t.year)

        if first_omitted:
            if month not in counts:
                counts[month] = 1
            else:
                counts[month] += 1

    counts.pop(month, None)

    total = sum(counts.values())
    count = len(counts)

    if count == 0:
        return 0
    return total / count


@lru_cache(maxsize=None)
def get_tweets_per_week(user_id):
    """returns the average number of tweets per week.
    First and last week in the database are omitted since it can't be assumed that they are complete."""
    times = sorted(db.get_post_times_of_user(user_id))
    counts = dict()
    week = ""
    first_omitted = False
    for t in times:
        # dont count first
        if week != str(t.isocalendar()[1]) + "_" + str(t.year) and week != "":
            first_omitted = True
        week = str(t.isocalendar()[1]) + "_" + str(t.year)

        if first_omitted:
            if week not in counts:
                counts[week] = 1
            else:
                counts[week] += 1

    counts.pop(week, None)

    total = sum(counts.values())
    count = len(counts)

    if count == 0:
        return 0
    return total / count


@lru_cache(maxsize=None)
def get_maximum_time_between_tweets(user_id):
    """calculates the maximum time between two tweets of a user"""
    times = sorted(db.get_post_times_of_user(user_id))
    max_diff = 0

    length = len(times)
    n = 2
    for i in range(length - n + 1):
        compare = list()
        for j in range(i, i + n):
            compare.append(times[j])
        diff = TimeUtils.time_diff_in_min(compare[1], compare[0])
        if diff > max_diff:
            max_diff = diff
    return max_diff


@lru_cache(maxsize=None)
def get_minimum_time_between_tweets(user_id):
    """calculates the minimum time between two tweets of a user"""
    times = sorted(db.get_post_times_of_user(user_id))
    min_diff = 0

    length = len(times)
    n = 2
    for i in range(length - n + 1):
        compare = list()
        for j in range(i, i + n):
            compare.append(times[j])
        diff = TimeUtils.time_diff_in_min(compare[1], compare[0])
        if diff > min_diff:
            min_diff = diff
    return min_diff


@lru_cache(maxsize=None)
def get_median_time_between_tweets(user_id):
    """calculates the median time between two tweets of a user"""
    times = sorted(db.get_post_times_of_user(user_id))

    diffs = list()

    length = len(times)
    n = 2
    for i in range(length - n + 1):
        compare = list()
        for j in range(i, i + n):
            compare.append(times[j])
        diffs.append(TimeUtils.time_diff_in_min(compare[1], compare[0]))

    if diffs:
        return median(sorted(diffs))
    else:
        return 0


@lru_cache(maxsize=None)
def get_avg_time_between_tweets(user_id):
    """calculates the avg time between two tweets of a user"""
    times = sorted(db.get_post_times_of_user(user_id))

    diffs = list()

    length = len(times)
    n = 2
    for i in range(length - n + 1):
        compare = list()
        for j in range(i, i + n):
            compare.append(times[j])
        diffs.append(TimeUtils.time_diff_in_min(compare[1], compare[0]))
    if diffs:
        return sum(diffs) / len(diffs)
    else:
        return 0


@lru_cache(maxsize=None)
def get_tweets_per_day(user_id):
    """returns the average number of tweets per day.
    First and last day in the database are omitted since it can't be assumed that they are complete."""
    times = sorted(db.get_post_times_of_user(user_id))
    counts = dict()
    day = ""
    first_omitted = False
    for t in times:
        # dont count first
        if day != str(t.day) + "_" + str(t.month) + "_" + str(t.year) and day != "":
            first_omitted = True
        day = str(t.day) + "_" + str(t.month) + "_" + str(t.year)

        if first_omitted:
            if day not in counts:
                counts[day] = 1
            else:
                counts[day] += 1

    counts.pop(day, None)

    total = sum(counts.values())
    count = len(counts)

    if count == 0:
        return 0
    return total / count


@lru_cache(maxsize=None)
def get_avg_user_mentions_per_tweet(user_id):
    """returns the average number of user mentions per tweet of user user_id"""
    return int(db.get_nr_of_user_mentions(user_id)) / len(db.get_tweets_of_user(user_id, ['id']))


@lru_cache(maxsize=None)
def get_avg_hashtags_per_tweet(user_id):
    """returns the average number of hashtags per tweet of user user_id"""
    return int(db.get_nr_of_hashtags(user_id)) / len(db.get_tweets_of_user(user_id, ['id']))


@lru_cache(maxsize=None)
def get_avg_urls_per_tweet(user_id):
    """returns the average number of urls per tweet of user user_id"""
    ids = db.get_tweets_of_user(user_id, ['entities_id', 'text'])
    sum = 0
    for id in ids:
        sum += tweet_nr_of_urls(id['entities_id'], id['text'])

    return sum / len(ids)


@lru_cache(maxsize=None)
def get_percent_with_url(user_id):
    """returns the percentage of tweets of user user_id that contains at least one url"""
    ids = db.get_tweets_of_user(user_id, ['entities_id', 'text'])
    sum = 0
    for id in ids:
        if tweet_contains_urls(id['entities_id'], id['text']):
            sum += 1

    return sum / len(ids)


@lru_cache(maxsize=None)
def get_percent_with_hashtag(user_id):
    """returns the percentage of tweets of user user_id that contains at least one hashtag"""
    ids = db.get_tweets_of_user(user_id, ['entities_id'])
    sum = 0
    for id in ids:
        if tweet_contains_hashtags(id['entities_id']):
            sum += 1

    return sum / len(ids)


@lru_cache(maxsize=None)
def get_percent_with_user_mention(user_id):
    """returns the percentage of tweets of user user_id that contains at least one user_mention"""
    ids = db.get_tweets_of_user(user_id, ['entities_id'])
    sum = 0
    for id in ids:
        if tweet_contains_user_mention(id['entities_id']):
            sum += 1

    return sum / len(ids)


@lru_cache(maxsize=None)
def get_nr_of_retweets_per_tweet(user_id):
    nr_of_retweets = db.get_nr_of_retweets_by_user(user_id)

    return nr_of_retweets / len(db.get_tweets_of_user(user_id, ['id']))


@lru_cache(maxsize=None)
def get_nr_of_replies_per_tweet(user_id):
    nr_of_replies = db.get_nr_of_replies_by_user(user_id)

    return nr_of_replies / len(db.get_tweets_of_user(user_id, ['id']))


@lru_cache(maxsize=None)
def get_nr_of_quotes_per_tweet(user_id):
    nr_of_quotes = db.get_nr_of_quotes_by_user(user_id)

    return nr_of_quotes / len(db.get_tweets_of_user(user_id, ['id']))


def get_possibly_sensitive(x):
    """returns the original value if it is not null, if there is a missing value, returns 2"""
    if pd.isnull(x):
        return 2
    return x


def tweet_nr_of_hashtags_in_popular_hashtags(entity_id, popular_hashtags):
    """returns the number of hashtags of a tweet that are in the most popular hashtags"""
    if pd.isnull(entity_id):
        return 0
    else:
        hashtags = db.get_hashtags_of_tweet(entity_id)
        count = 0
        for h in popular_hashtags:
            if h in hashtags:
                count += 1
        return count


def slang_words_in_tweet(tokens):
    """returns a list with the slang words found in the tweet"""
    slang = NLPUtils.get_slang_words()
    return [t for t in tokens if t in slang]


def tweet_avg_word_length(tokens):
    """returns the avg word length of a tokenized sentence"""
    sum = 0
    for t in tokens:
        t = TextPreprocessor.replace_all_punctuation(t)
        if t != "":
            if t[0] != '#' and t[0] != '@' and t[0] != '$' and t != '' and t.lower() != 'rt':
                sum += len(t)
    if len(tokens) == 0:
        return 0
    else:
        return sum / len(tokens)


def nr_of_capitalized_words(tokens):
    """returns the number of capitalized words. All capitalized words do not count"""
    count = 0
    for t in tokens:
        # do not count words like 'I'
        if len(t) > 1:
            if t[0].isupper():
                for i in range(1, len(t)):
                    if t[i].isupper():
                        break
                else:
                    count += 1
    return count


def nr_of_all_capitalized_words(tokens):
    """returns all words with only uppercase letters"""
    count = 0
    for t in tokens:
        if t != '':
            t = TextPreprocessor.replace_all_punctuation(t)
            if t != "":
                if t[0] != '#' and t[0] != '@' and t[0] != '$' and t != '' and t.lower() != 'rt':
                    for i in range(len(t)):
                        if t[i].islower():
                            break
                    else:
                        count += 1
    return count


def tweet_find_stock_mention(text):
    """finds stock mentions in the text (all words that start with $)"""
    stocks = list()
    tokens = text.split()
    for t in tokens:
        if len(t) > 0:
            if re.match("\$\w+", t):
                stocks.append(t)
    return stocks


def nr_of_cased_characters(tokens):
    """returns the number of alphabet characters in a tokenized tweet"""
    count = 0
    for token in tokens:
        for t in token:
            if t.isupper() or t.islower():
                count += 1
    return count


def tweet_count_upper_letters(tokens):
    """returns the number of upper case letters"""
    count = 0
    for token in tokens:
        for t in token:
            if t.isupper():
                count += 1
    return count


def contains_all_uppercase(text):
    """finds sequences of at least 5 uppercase characters."""
    text = TextPreprocessor.remove_user_mentions(text)
    text = TextPreprocessor.remove_hashtags(text)
    if re.findall(r'([A-Z]+[!.,]?(.)?){5,}', text):
        return True
    else:
        return False


def is_upper(text):
    """removes urls, hashtags and user mentions, then checks for isupper()"""
    text = TextPreprocessor.remove_urls(text)
    text = TextPreprocessor.remove_hashtags(text)
    text = TextPreprocessor.remove_user_mentions(text)
    return text.isupper()


def get_tag_ratio(tagged_text, tag, nr_of_words):
    """returns the ratio of tokens with a specific tag.
    tag: N: Noun, A: Adjective, V: Verb"""

    tagged = json.loads(tagged_text)
    a_count = 0
    for t in tagged:
        if t['tag'] == tag:
            token = TextPreprocessor.replace_all_punctuation(t['token'])
            if token != "":
                if token[0] != '#' and token[0] != '@' and token[0] != '$' and token != '' and token.lower() != 'rt':
                    a_count += 1
    if nr_of_words == 0:
        return 0
    else:
        return a_count / nr_of_words


def tweet_contains_character_repetitions(text):
    # look for a character followed by at least two repetition of itself.
    pattern = re.compile(r"(.)\1{2,}")
    if re.findall(pattern, text):
        return True
    else:
        return False


def get_nr_of_words(tokens):
    "counts all tokens except punctuation, user mentions, stock or hashtags"
    count = 0
    for t in tokens:
        if t != '':
            t = TextPreprocessor.replace_all_punctuation(t)
            if t != "":
                if t[0] != '#' and t[0] != '@' and t[0] != '$' and t != '' and t.lower() != 'rt':
                    count += 1
    return count


def ratio_punctuation_tokens(tokenized_text, nr_of_tokens):
    if nr_of_tokens == 0:
        return 0
    else:
        tokens = NLPUtils.str_list_to_list(tokenized_text)

        punctuation = NLPUtils.get_punctuation()
        count = 0
        for token in tokens:
            if token in punctuation:
                count += 1

        return count / nr_of_tokens


def get_top_level_domain_of_expanded_url(url):
    """looks up the expanded versions of the url and returns the top level domain of the most expanded"""
    if url is not None:
        url = db.get_url(url)

        manual_expanded_url = url['manual_expanded_url']
        expanded_url = url['expanded_url']

        if manual_expanded_url is not None:
            return UrlUtils.get_top_level_domain(manual_expanded_url)
        elif expanded_url is not None:
            return UrlUtils.get_top_level_domain(expanded_url)
        else:
            return UrlUtils.get_top_level_domain(url)
    else:
        return ""


def tweet_contains_named_entities(pos_tags):
    """returns True if the tweets contain at 
    least one token that is tagged as an named entity"""
    for token in pos_tags:
        if token["tag"] == '^':
            return True
    return False


def get_nr_of_punctuation(text):
    """counts punctuation"""
    cnt = Counter()
    punctuation = NLPUtils.get_punctuation()
    for t in text:
        if t in punctuation:
            cnt[t] += 1
    return cnt


def tweet_ratio_tokens_before_after_prepro(tokens_before, tokens_after):
    """calculates the ratio of the number of tokens before 
    to the number of tokens after additional preprocessing"""
    if not tokens_before:
        return 0
    else:
        return len(tokens_after) / len(tokens_before)


def tweet_quarter(month):
    """returns the quarter of the year"""
    if month <= 3:
        return 0
    elif 3 < month <= 6:
        return 1
    elif 6 < month <= 9:
        return 2
    elif 9 < month:
        return 3


def remove_um_url(tokens):
    """
    replaces urls and user mentions
    :param tokens: 
    :return: 
    """
    new = list()
    for token in tokens:
        token = TextPreprocessor.remove_urls(token)
        token = TextPreprocessor.remove_user_mentions(token)
        token = TextPreprocessor.remove_hashtags(token)
        if token != "":
            new.append(token)
    return str(new)


def contains_number(tokens):
    """
    True, if tweet contains a token that is a number
    :param tokens: 
    :return: 
    """
    for token in tokens:
        if re.match('^\d+$', token):
            return True
    return False


def contains_quote(text):
    """
    finds quotes in a text
    :param text: 
    :return: 
    """
    res = re.findall('"([^"]*)"', text)
    if res:
        return True
    else:
        return False


def tweet_ratio_sentiment_words(pos_neg, nr_sent_words):
    if nr_sent_words == 0:
        return 0
    else:
        return pos_neg / nr_sent_words

