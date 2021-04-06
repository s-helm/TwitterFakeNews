
import collections
from functools import lru_cache

import pandas as pd
import pymysql
from tweepy.streaming import json

from Domain.ExtendedEntity import ExtendedEntity
from Domain.Media import Media
from Learning.LearningUtils import get_dataset, get_testset
from Utility.AccountFilesHandler import get_accounts
from Utility.Constants import REAL
from Utility.UrlUtils import UrlUtils
from Utility.Util import generate_alias_select_sting


class DatabaseHandler:
    import re

    db = pymysql.connect(host='localhost', user='root', password='******', db='twitterschema', charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)

    db_trending_topics = None


    prog = re.compile('(__.*__)')

    START_DATE = '2017-02-01 00:00:00'

    ACCOUNTS_JOIN_TABLE = 'user_crawled_original'
    # ACCOUNTS_JOIN_TABLE = 'user_crawled'
    WHERE = " AND tweet.retweeted_status_id is null " \
            "AND tweet.quoted_status_id is null " \
            "AND tweet.in_reply_to_status_id is null " \

    source_sel = ["fake_opensource",
                  # "satire_opensource",
                  "fake_research",
                  # "parody_research",
                  "reliable_opensource",
                  # "reliable_dmoz",
                  "reliable_dmoz_local",
                  "reliable_study"]
    source_sel_query = " OR ".join(["source = \"" + src + "\"" for src in source_sel])
    USER_SELECTION = "(SELECT DISTINCT name FROM " + ACCOUNTS_JOIN_TABLE + " " \
                        "WHERE " + source_sel_query + ") "

    @staticmethod
    def connect_to_trending_topics_db():
        DatabaseHandler.db_trending_topics = pymysql.connect(host='localhost', user='root', password='******', db='trending_topics',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


    @staticmethod
    def get_worldwide_trends_by_date():
        """
        fetches the worldwide trending topics
        :return: worldwide trending topics
        """
        query = "SELECT hashtag, as_of FROM trend t, location l " \
                 "WHERE l.id = t.location_id " \
                 "AND l.woeid = 1;"

        cur = DatabaseHandler.db_trending_topics.cursor()
        cur.execute(query)
        DatabaseHandler.db_trending_topics.commit()
        result = cur.fetchall()
        cur.close()

        trends = dict()

        for res in result:
            date = res['as_of'].date()
            if date in trends:
                trends[date].append(res['hashtag'])
            else:
                trends[date] = [res['hashtag']]
        return trends

    @staticmethod
    def get_trends_by_date_and_woeid():
        """
        fetches the worldwide trending topics
        :return: worldwide trending topics
        """
        query = "SELECT t.hashtag, t.as_of, l.woeid FROM trend t, location l " \
                 "WHERE l.id = t.location_id;"

        cur = DatabaseHandler.db_trending_topics.cursor()
        cur.execute(query)
        DatabaseHandler.db_trending_topics.commit()
        result = cur.fetchall()
        cur.close()

        trends = dict()

        for res in result:
            date = res['as_of'].date()
            woeid = res['woeid']
            if date in trends:
                if woeid in trends[date]:
                    trends[date][woeid].append(res['hashtag'])
                else:
                    trends[date][woeid] = [res['hashtag']]
                    # trends[date].append(res['hashtag'])
            else:
                trends[date] = dict()
                trends[date][woeid] = [res['hashtag']]
        return trends

    @staticmethod
    def get_most_popular_hashtags_across_users(nr_of_hashtags):
        """returns the hashtags that were used by the most accounts
        -nr_of_hashtags: the number of hashtags that should be returned"""
        query = "SELECT count(distinct user.id) as c, eh.hashtag " \
                "From user, " \
                    "(SELECT user_id, created_at, entities_id, lang " \
                    "FROM tweet " \
                    "WHERE lang = 'en' " \
                    + DatabaseHandler.WHERE + \
                    "AND created_at between %s and now()) as t, " \
                "entity e, entity_hashtag eh " \
                "WHERE lcase(user.screen_name) IN "+DatabaseHandler.USER_SELECTION+" " \
                "AND t.user_id = user.id " \
                "AND t.entities_id = e.id " \
                "AND e.id = eh.entity_id " \
                "GROUP BY eh.hashtag " \
                "ORDER BY c desc " \
                "LIMIT %s;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, (DatabaseHandler.START_DATE, nr_of_hashtags))
        DatabaseHandler.db.commit()
        most_frequent = [item['hashtag'] for item in cur.fetchall()]
        cur.close()
        return most_frequent

    @staticmethod
    def get_tweets_with_key(attrs):
        """returns a tweets attritubutes and its id"""
        select = ""
        for a in attrs:
            select += ", "+a
        query = "SELECT key_id"+select+" FROM tweet;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        tweets = cur.fetchall()
        cur.close()
        return tweets

    @staticmethod
    def get_tweets_without_additional_preprocessed_text():
        """returns all tweets that have not been preprocessed yet"""
        # query = "SELECT key_id, pos_tags FROM tweet WHERE additional_preprocessed_text is null;"
        query = "SELECT key_id, pos_tags FROM tweet;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        tweets = cur.fetchall()
        cur.close()
        return tweets

    @staticmethod
    def get_tweets_without_feature(null_feature, select_feature="text", new_only=True, testset=False):
        """returns all tweets that have not been sentence tokenized yet"""
        if testset:
            if type(select_feature) == list:
                select_feature = ', '.join(select_feature)

            if new_only and DatabaseHandler.check_column_exists('tweet', null_feature):
                query = "SELECT key_id, " + select_feature + " FROM tweet, testset WHERE " + null_feature + " is null AND testset.tweet_id = tweet.id;"
            else:
                query = "SELECT key_id, " + select_feature + " FROM tweet, testset WHERE testset.tweet_id = tweet.id;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(query)
            DatabaseHandler.db.commit()
            tweets = cur.fetchall()
            cur.close()
            return tweets
        else:
            if type(select_feature) == list:
                select_feature = ', '.join(select_feature)

            if new_only and DatabaseHandler.check_column_exists('tweet', null_feature):
                query = "SELECT key_id, "+select_feature+" FROM tweet, user u, user_crawled_original uc " \
                                                         "WHERE tweet.user_id = u.id " \
                                                         "AND lcase(u.screen_name) = uc.name " \
                                                         "AND "+null_feature+" is null;"
            else:
                query = "SELECT key_id, "+select_feature+" FROM tweet, user u, user_crawled_original uc " \
                                                         "WHERE tweet.user_id = u.id " \
                                                         "AND lcase(u.screen_name) = uc.name;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(query)
            DatabaseHandler.db.commit()
            tweets = cur.fetchall()
            cur.close()
            return tweets

    @staticmethod
    def get_tweets_without_features(null_feature, select_features, new_only=True, testset=False):
        """returns all tweets that have not been sentence tokenized yet"""
        if testset:
            select = ", ".join(select_features)
            if new_only and DatabaseHandler.check_column_exists('tweet', null_feature):
                query = "SELECT key_id, " + select + " FROM tweet, testset WHERE " + null_feature + " is null and testset.tweet_id = tweet.id;;"
            else:
                query = "SELECT key_id, " + select + " FROM tweet, testset WHERE testset.tweet_id = tweet.id;;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(query)
            DatabaseHandler.db.commit()
            tweets = cur.fetchall()
            cur.close()
            return tweets
        else:
            select = ", ".join(select_features)
            if new_only and DatabaseHandler.check_column_exists('tweet', null_feature):
                query = "SELECT key_id, "+select+" FROM tweet WHERE "+null_feature+" is null;"
            else:
                query = "SELECT key_id, "+select+" FROM tweet;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(query)
            DatabaseHandler.db.commit()
            tweets = cur.fetchall()
            cur.close()
            return tweets

    @staticmethod
    def insert_sent_tokenized_text(insert_collection):
        """inserts the tokenized text in into the database"""
        query = "UPDATE tweet SET sent_tokenized_text=%s WHERE key_id=%s;"
        cur = DatabaseHandler.db.cursor()
        cur.executemany(query, (insert_collection))
        DatabaseHandler.db.commit()
        cur.close()

    @staticmethod
    def insert_tokenized_text(key_id, tokenized_text):
        """inserts the tokenized text in into the database"""
        query = "UPDATE tweet SET tokenized_text=%s WHERE key_id=%s;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query, (str(tokenized_text), key_id))
        DatabaseHandler.db.commit()
        cur.close()

    @staticmethod
    def remove_tweets_from_users_not_used():
        """
        returns all user names which are not used and deletes them and their respective tweets
        :return: 
        """
        query = "SELECT u.id FROM user u WHERE lcase(u.screen_name) NOT IN (SELECT uc.name FROM user_crawled_original uc);"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        users_to_remove = [item['id'] for item in cur.fetchall()]
        cur.close()

        print("{} users to remove...".format(len(users_to_remove)))
        for user_id in users_to_remove:
            print("Delete tweets from user: {}".format(user_id))
            query = "DELETE FROM tweet WHERE user_id = %s;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(query, user_id)
            DatabaseHandler.db.commit()
            cur.close()

    @staticmethod
    def remove_users_not_used():
        query = "SELECT u.id FROM user u WHERE lcase(u.screen_name) NOT IN (SELECT uc.name FROM user_crawled_original uc);"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        users_to_remove = [item['id'] for item in cur.fetchall()]
        cur.close()

        print("{} users to remove...".format(len(users_to_remove)))
        for user_id in users_to_remove:
            print("Delete user: {}".format(user_id))
            query = "DELETE FROM user WHERE id = %s;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(query, user_id)
            DatabaseHandler.db.commit()
            cur.close()


    @staticmethod
    def remove_account(name):
        """removes account with screen_name 'name' and all its tweets.
        Returns the number of removed tweets"""
        name = name.lower()

        tweets_query = "SELECT t.key_id FROM tweet t, user u WHERE t.user_id = u.id AND lcase(u.screen_name) = %s;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(tweets_query, name)
        DatabaseHandler.db.commit()
        tweet_ids = [item['key_id'] for item in cur.fetchall()]
        cur.close()

        acc_query = "SELECT id FROM user WHERE lcase(screen_name) = %s;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(acc_query, name)
        DatabaseHandler.db.commit()
        res = cur.fetchone()
        cur.close()
        acc_id = None
        if res is not None:
            acc_id = res['id']

        delete_tweet = "DELETE FROM tweet WHERE key_id = %s;"

        for id in tweet_ids:
            cur = DatabaseHandler.db.cursor()
            cur.execute(delete_tweet, id)
            DatabaseHandler.db.commit()
            cur.close()
        if acc_id is not None:
            delete_acc = "DELETE FROM user WHERE id = %s;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(delete_acc, acc_id)
            DatabaseHandler.db.commit()
            cur.close()

        delete_user_crawled = "DELETE FROM user_crawled WHERE lcase(name) = %s;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(delete_user_crawled, name)
        DatabaseHandler.db.commit()
        cur.close()
        print("{} tweets removed for account {}".format(len(tweet_ids),name))
        return len(tweet_ids)

    @staticmethod
    def remove_non_english_tweets():
        """removes all tweets that are not english language"""
        query = "SELECT key_id FROM tweet WHERE lang!='en' AND lang!='und';"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        ids = [item['key_id'] for item in cur.fetchall()]
        cur.close()

        delete = "DELETE FROM tweet WHERE key_id=%s;"
        cur = DatabaseHandler.db.cursor()
        cur.executemany(delete, ids)
        DatabaseHandler.db.commit()
        cur.close()

        print("{} tweets removed.".format(len(ids)))


    @staticmethod
    def find_accounts_with_major_lang_not_english():
        """finds all accounts that do not post at least 50% of their tweets in english language"""
        query = "SELECT count(tweet.id) as c, tweet.lang, lcase(user.screen_name) as screen_name " \
                "FROM user, tweet " \
                "WHERE lcase(user.screen_name) IN " + DatabaseHandler.USER_SELECTION + " " \
                "AND tweet.user_id = user.id " \
                "GROUP BY tweet.lang, user.screen_name;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        res_dict = cur.fetchall()
        cur.close()

        users = DatabaseHandler.get_users_to_crawl()

        counts = collections.defaultdict(dict)
        for item in res_dict:
                if item['lang'] == 'en':
                    counts[item['screen_name']][item['lang']] = item['c']
                else:
                    if item['screen_name'] in counts and 'other' in counts[item['screen_name']]:
                        counts[item['screen_name']]['other'] += item['c']
                    else:
                        counts[item['screen_name']]['other'] = item['c']

        en_accs = list()
        non_en_accs = list()
        for u in users:
            other_count = 0
            en_count = 0
            if 'en' in counts[u]:
                en_count = counts[u]['en'];

            if 'other' in counts[u]:
                other_count = counts[u]['other']

            if en_count == en_count == 0:
                non_en_accs.append(u)
                print("User: " + u + " en: " + str(en_count) + " other: " + str(other_count)  + " (-> OTHER)")
            # at least 2/3 of the tweets needs to be english
            elif (en_count)/(other_count+en_count) > 0.66:
                en_accs.append(u)
                print("User: " + u + " en: " + str(en_count) + " other: " + str(other_count) + " (-> EN)")
            else:
                non_en_accs.append(u)
                print("User: " + u + " en: " + str(en_count) + " other: " + str(other_count)  + " (-> OTHER)")

        with open('../accounts/english_accounts.json', 'w') as outfile:
            json.dump(en_accs, outfile)

        with open('../accounts/non_english_accounts.json', 'w') as outfile:
            json.dump(non_en_accs, outfile)

    @staticmethod
    def get_user_lang_counts(user_id):
        """returns the languages of tweets by a user. 'und' is not counted"""
        query = "SELECT count(t.lang) as lang_count, t.lang FROM tweet t " \
                "WHERE t.user_id = %s " \
                "GROUP BY t.lang;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query, user_id)
        DatabaseHandler.db.commit()
        langs = cur.fetchall()
        cur.close()
        return langs

    @staticmethod
    def get_users_to_crawl():
        """returns all user screen_names from user_crawled"""
        user_query = "SELECT lcase(uc.name) as name FROM "+ DatabaseHandler.ACCOUNTS_JOIN_TABLE + " uc"
        cur = DatabaseHandler.db.cursor()
        cur.execute(user_query)
        DatabaseHandler.db.commit()
        users = [item['name'] for item in cur.fetchall()]
        cur.close()
        return users

    @staticmethod
    @lru_cache(maxsize=None)
    def get_nr_of_retweets_by_user(user_id):
        """returns the number of tweets this user_id has retweeted"""
        query = "SELECT count(retweeted_status_id) as count From tweet " \
                "WHERE tweet.user_id = %s"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, user_id)
        DatabaseHandler.db.commit()
        count = cur.fetchone()['count']
        cur.close()
        return count

    @staticmethod
    @lru_cache(maxsize=None)
    def get_nr_of_quotes_by_user(user_id):
        """returns the number of tweets this user_id has quoted"""
        query = "SELECT count(quoted_status_id) as count From tweet " \
                "WHERE tweet.user_id = %s"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, user_id)
        DatabaseHandler.db.commit()
        count = cur.fetchone()['count']
        cur.close()
        return count

    @staticmethod
    @lru_cache(maxsize=None)
    def get_nr_of_replies_by_user(user_id):
        """returns the number of tweets this user_id has quoted"""
        query = "SELECT count(in_reply_to_status_id) as count From tweet " \
                "WHERE tweet.user_id = %s"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, user_id)
        DatabaseHandler.db.commit()
        count = cur.fetchone()['count']
        cur.close()
        return count

    @staticmethod
    def get_tweets_of_user(user_id, attrs):
        """selects all the attribute attr of all tweets of a user"""

        select = ', '.join(attrs)
        query = "SELECT " + select + " " \
                "FROM tweet " \
                "WHERE user_id = %s;" \

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, user_id)
        DatabaseHandler.db.commit()
        res = cur.fetchall()
        cur.close()
        return res

    @staticmethod
    @lru_cache(maxsize=None)
    def get_nr_of_user_mentions(user_id):
        """returns the total number of user mentions in a users tweets"""
        query = "SELECT count(eu.user_mentions_id) as count " \
                "FROM tweet, entity, entity_user_mentions eu " \
                "WHERE tweet.user_id = %s " \
                "AND tweet.entities_id = entity.id " \
                "AND entity.id = eu.entity_id"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, user_id)
        DatabaseHandler.db.commit()
        count = cur.fetchone()['count']
        cur.close()
        return count

    @staticmethod
    @lru_cache(maxsize=None)
    def get_nr_of_hashtags(user_id):
        query = "SELECT count(eh.hashtag) as count FROM tweet, entity, entity_hashtag eh " \
                "WHERE tweet.user_id = %s " \
                "AND tweet.entities_id = entity.id " \
                "AND entity.id = eh.entity_id;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, user_id)
        DatabaseHandler.db.commit()
        count = cur.fetchone()['count']
        cur.close()
        return count

    @staticmethod
    def get_user_selection(attrs):
        """returns a list with all users and their attributes. Default: returns only ids"""
        select = ""
        for attr in attrs:
            select += ", u." + attr
        query = "SELECT u.id "+select+" FROM user u " \
                "WHERE lcase(u.screen_name) IN " + DatabaseHandler.USER_SELECTION + ";"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        list = cur.fetchall()
        cur.close()
        return list

    @staticmethod
    def insert_user_feature(user_id, feature_name, value, sql_type):
        """checks if a column exists for the feature, otherwise it creates a column for it"""

        if DatabaseHandler.check_column_exists('user', feature_name):
            query = "UPDATE user SET " + feature_name + "=%s WHERE id=%s;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(query, (value, user_id))
            DatabaseHandler.db.commit()
            cur.close()
        else:
            DatabaseHandler.create_feature_in_table('user', feature_name, sql_type)
            query = "UPDATE user SET " + feature_name + "=%s WHERE id=%s;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(query, (value, user_id))
            DatabaseHandler.db.commit()
            cur.close()

    @staticmethod
    def insert_tweet_features(feature_names, values, sql_types):
        """
        Inserts multiple features. Creates column if not exist. 
        :param feature_names: list with names of the features to insert
        :param values: list of tuples that contain the values. Last index of tuple has to contain the tweet key_id
        :param sql_types: contains a list with the sql types of the attributes (only necessary if column does not exist yet)
        :return: -
        """
        for i in range(len(feature_names)):
            if not DatabaseHandler.check_column_exists('tweet', feature_names[i]):
                DatabaseHandler.create_feature_in_table('tweet', feature_names[i], sql_types[i])

        feature_names_str = [f + '=%s' for f in feature_names]
        query = "UPDATE tweet SET " + ', '.join(feature_names_str) + " WHERE key_id=%s;"
        cur = DatabaseHandler.db.cursor()
        cur.executemany(query, values)
        DatabaseHandler.db.commit()
        cur.close()


    @staticmethod
    def insert_tweet_feature(tweet_id, feature_name, value, sql_type):
        """checks if a column exists for the feature, otherwise it creates a column for it"""

        if not DatabaseHandler.check_column_exists('tweet', feature_name):
            DatabaseHandler.create_feature_in_table('tweet', feature_name, sql_type)

        if type(value) == list:
            query = "UPDATE tweet SET " + feature_name + "=%s WHERE key_id=%s;"
            cur = DatabaseHandler.db.cursor()
            cur.executemany(query, value)
            DatabaseHandler.db.commit()
            cur.close()
        else:
            query = "UPDATE tweet SET " + feature_name + "=%s WHERE key_id=%s;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(query, (value, tweet_id))
            DatabaseHandler.db.commit()
            cur.close()

    @staticmethod
    def check_column_exists(table, column):
        """checks if a column exists in a database table"""
        query = "SHOW COLUMNS FROM "+table+" LIKE %s;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query, column)
        DatabaseHandler.db.commit()
        exists = cur.fetchone()
        cur.close()
        return exists


    @staticmethod
    def create_feature_in_table(table, name, sql_type):
        query = "ALTER TABLE "+table+" ADD COLUMN "+name+" "+sql_type+" NULL DEFAULT NULL;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        cur.close()

    @staticmethod
    def get_users(all=False):
        """return all user accounts that where directly crawled (does not include users from retweets"""
        if all:
            query = "SELECT user.* FROM user, user_crawled_original uc " \
                    "WHERE lcase(user.screen_name) = uc.name;"
        else:
            query = "SELECT user.* FROM user " \
                    "WHERE lcase(user.screen_name) IN "+DatabaseHandler.USER_SELECTION+";"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        list = cur.fetchall()
        cur.close()
        return list

    @staticmethod
    def get_user_ids():
        query = "SELECT user.id FROM user " \
                "WHERE lcase(user.screen_name) IN "+DatabaseHandler.USER_SELECTION+";"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        list = [item['id'] for item in cur.fetchall()]
        cur.close()
        return list

    @staticmethod
    def get_tweets(attr, all=False):
        """returns attr of all tweeds as list that match the training set criteria"""
        select = None
        if type(attr) == list:
            select = ', '.join(attr)
        else:
            select = attr

        cur = DatabaseHandler.db.cursor()
        if all:
            query = "SELECT "+select+" FROM user, tweet " \
                    "WHERE tweet.user_id = user.id " \
                    "AND tweet.lang = 'en';"
            cur.execute(query)
        else:
            query = "SELECT "+select+" FROM user, tweet " \
                    "WHERE lcase(user.screen_name) IN "+DatabaseHandler.USER_SELECTION+" "  \
                    "AND tweet.user_id = user.id " \
                    "AND tweet.created_at between %s and now() " \
                    + DatabaseHandler.WHERE + \
                    "AND tweet.lang = 'en';"
            cur.execute(query, DatabaseHandler.START_DATE)

        DatabaseHandler.db.commit()
        res = cur.fetchall()
        cur.close()
        if type(attr) == list:
            return res
        else:
            return [item[attr] for item in res]

    @staticmethod
    def get_hashtags_of_tweet(entity_id):
        """returns all hashtags of a tweet"""
        query = "SELECT eh.hashtag FROM entity_hashtag eh " \
                "WHERE eh.entity_id = %s;" \

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, entity_id)
        DatabaseHandler.db.commit()
        hashtags = [item['hashtag'] for item in cur.fetchall()]
        cur.close()
        return hashtags

    @staticmethod
    def get_urls_of_tweet(entity_id):
        """returns all urls of a tweet"""
        query = "SELECT eu.url_url FROM entity_url eu " \
                "WHERE eu.entity_id = %s;" \

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, entity_id)
        DatabaseHandler.db.commit()
        urls = [item['url_url'] for item in cur.fetchall()]
        cur.close()
        return urls

    @staticmethod
    def get_manual_expanded_urls_of_tweet(entity_id):
        """returns all urls of a tweet as an expanded url"""
        query = "SELECT u.manual_expanded_url " \
                "FROM entity_url eu, url u " \
                "WHERE eu.entity_id = %s " \
                "AND u.url = eu.url_url;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, entity_id)
        DatabaseHandler.db.commit()
        urls = [item['manual_expanded_url'] for item in cur.fetchall()]
        cur.close()
        return urls

    @staticmethod
    @lru_cache(maxsize=None)
    def get_user_location(user_id):
        query = "SELECT location " \
                "FROM user " \
                "WHERE id = %s;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, user_id)
        DatabaseHandler.db.commit()
        tz = cur.fetchone()['location']
        cur.close()
        return tz


    @staticmethod
    @lru_cache(maxsize=None)
    def get_user_utc_offset(user_id):
        query = "SELECT utc_offset " \
                "FROM user " \
                "WHERE id = %s;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, user_id)
        DatabaseHandler.db.commit()
        tz = cur.fetchone()['utc_offset']
        cur.close()
        return tz

    @staticmethod
    @lru_cache(maxsize=None)
    def get_post_times_of_user(user_id):
        query = "SELECT t.created_at " \
                "FROM tweet t " \
                "WHERE t.user_id = %s;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, user_id)
        DatabaseHandler.db.commit()
        dates = [item['created_at'] for item in cur.fetchall()]
        cur.close()
        return dates

    @staticmethod
    def get_urls(entity_id):
        query = "SELECT u.url, u.expanded_url, u.manual_expanded_url FROM entity_url eu, url u " \
                "WHERE u.url = eu.url_url " \
                "AND eu.entity_id = %s;" \

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, entity_id)
        DatabaseHandler.db.commit()
        urls = cur.fetchall()
        cur.close()
        return urls

    @staticmethod
    def get_url(url):
        query = "SELECT u.url, u.expanded_url, u.manual_expanded_url FROM url u " \
                "WHERE u.url = %s";

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, url)
        DatabaseHandler.db.commit()
        urls = cur.fetchone()
        cur.close()
        return urls

    @staticmethod
    def get_manual_expanded_url_by_url(url):
        """returns the expanded url from the database"""
        query = "SELECT manual_expanded_url FROM url " \
                "WHERE url = %s;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, url)
        DatabaseHandler.db.commit()
        urls = [item['manual_expanded_url'] for item in cur.fetchall()]
        cur.close()
        return urls

    @staticmethod
    def get_user_mentions(entity_id):
        """returns all user mentions of a tweet"""
        query = "SELECT eu.user_mentions_id FROM entity_user_mentions eu " \
                "WHERE eu.entity_id = %s;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, entity_id)
        DatabaseHandler.db.commit()
        user_mentions = [item['user_mentions_id'] for item in cur.fetchall()]
        cur.close()
        return user_mentions

    @staticmethod
    def get_media(entity_id):
        """returns all media of a tweet"""
        query = "SELECT em.media_id FROM entity_media em " \
                "WHERE em.entity_id = %s;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, entity_id)
        DatabaseHandler.db.commit()
        user_mentions = [item['media_id'] for item in cur.fetchall()]
        cur.close()
        return user_mentions

    @staticmethod
    def get_tweet_by_entities_id(attr, entity_id):
        """get tweet attribute attr by entities_id"""
        query = "SELECT %s FROM tweet " \
                "WHERE tweet.entities_id = %s;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, (attr, entity_id))
        DatabaseHandler.db.commit()
        urls = [item[attr] for item in cur.fetchall()]
        cur.close()
        return urls

    @staticmethod
    def load_data_set(columns=None, testset=False):
        """returns the data set"""
        tweet_cols = DatabaseHandler.get_column_names("tweet")
        user_cols = DatabaseHandler.get_column_names("user")

        list1 = generate_alias_select_sting(tweet_cols, "tweet")
        list2 = generate_alias_select_sting(user_cols, "user")
        list1.extend(list2)

        selection = list()
        if columns is not None:
            for col_prep in list1:
                found = False
                for col in columns:
                    if col in col_prep:
                        found = True
                if found:
                    selection.append(col_prep)

        if not selection:
            selection = list1
        select_string = ",".join(selection)

        if testset:
            print("Load testset...")

            query = "SELECT " + select_string + " " \
                                                "FROM user, tweet " \
                                                "WHERE lcase(tweet.id) IN (SELECT tweet_id FROM testset) " \
                                                "AND tweet.user_id = user.id;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(query)
            DatabaseHandler.db.commit()
            tweets = cur.fetchall()
            cur.close()
        else:
            print("Load dataset...")

            query = "SELECT " + select_string + " " \
                                                "FROM user, tweet " \
                                                "WHERE lcase(user.screen_name) IN "+DatabaseHandler.USER_SELECTION+" " \
                                               "AND tweet.user_id = user.id " \
                                               "AND tweet.created_at between %s and now() " \
                                               "AND tweet.lang = 'en' " \
                    + DatabaseHandler.WHERE

            cur = DatabaseHandler.db.cursor()
            cur.execute(query, DatabaseHandler.START_DATE)
            DatabaseHandler.db.commit()
            tweets = cur.fetchall()
            cur.close()

        return tweets

    @staticmethod
    def get_column_names(table_name):
        """returns a list with the table names of a column"""
        col_query = "SELECT COLUMN_NAME " \
                    "FROM INFORMATION_SCHEMA.COLUMNS " \
                    "WHERE TABLE_SCHEMA='twitterschema' " \
                    "AND TABLE_NAME=%s;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(col_query, table_name)
        DatabaseHandler.db.commit()
        cols = cur.fetchall()
        cur.close()

        return [col['COLUMN_NAME'] for col in cols]

    @staticmethod
    def get_all_tweet_ids():
        query = "SELECT t.id FROM tweet t"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        ids = [item['id'] for item in cur.fetchall()]
        cur.close()
        return ids

    @staticmethod
    def check_correct_labeling():
        """relables tweets of non fake news which were labeled as fake due to a retweet of a fake news accounts"""
        real_accounts = get_accounts(REAL)

        for acc in real_accounts:
            query = "SELECT t.key_id, u.screen_name FROM tweet t, user u " \
                    "WHERE t.user_id = u.id AND lcase(u.screen_name) = %s" \
                    "AND t.fake = 1"

            cur = DatabaseHandler.db.cursor()
            cur.execute(query, acc)
            DatabaseHandler.db.commit()
            tweets = cur.fetchall()
            cur.close()

            for t in tweets:
                screen_name = t['screen_name']
                key_id = t['key_id']

                print("Relabel tweet " + str(key_id) + " of \'" + screen_name + "\'")
                update = "UPDATE tweet SET fake = %s WHERE key_id = %s;"
                cur = DatabaseHandler.db.cursor()
                cur.execute(update, (0, key_id))
                DatabaseHandler.db.commit()
                cur.close()

    @staticmethod
    def remove_duplicate_tweets():
        query = "(SELECT key_id, id FROM tweet) "

        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        tweets = cur.fetchall()
        cur.close()

        list = []

        for t in tweets:
            id = t['id']
            key_id = t['key_id']
            if id in list:
                print("delete tweet with id: " + id + "(key_id: " + key_id + ")")
                insert_query = """DELETE FROM tweet WHERE key_id = %s;"""
                cur = DatabaseHandler.db.cursor()
                cur.execute(insert_query, key_id)
                DatabaseHandler.db.commit()
                cur.close()
            else:
                list.append(id)

    @staticmethod
    def clear_user_crawled_original():
        query = "TRUNCATE TABLE user_crawled_original"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        cur.close()

    @staticmethod
    def insert_users_crawled(accs):

        for acc in accs:
            DatabaseHandler.insert_user_crawled(acc)


    @staticmethod
    def insert_user_crawled_original(acc):
        """fills a second temporary table"""

        query = "(SELECT * FROM user_crawled_original WHERE lcase(name) = %s) "

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, acc.lower())
        DatabaseHandler.db.commit()
        is_in_db = cur.fetchone()
        cur.close()

        if is_in_db is None:
            insert_query = """INSERT INTO user_crawled_original (name)
                                                         VALUES(%s);"""
            cur = DatabaseHandler.db.cursor()
            cur.execute(insert_query, acc.lower())
            DatabaseHandler.db.commit()
            cur.close()

            print("Crawled account \"" + acc + "\" inserted into temporary table.")

    @staticmethod
    def insert_user_crawled(acc):

        query = "(SELECT * FROM user_crawled WHERE lcase(name) = %s) "

        cur = DatabaseHandler.db.cursor()
        cur.execute(query, acc.lower())
        DatabaseHandler.db.commit()
        is_in_db = cur.fetchone()
        cur.close()

        if is_in_db is None:
            insert_query = """INSERT INTO user_crawled (name)
                                                        VALUES(%s);"""
            cur = DatabaseHandler.db.cursor()
            cur.execute(insert_query, acc.lower())
            DatabaseHandler.db.commit()
            cur.close()

            print("Crawled account \"" + acc + "\" inserted.")

    @staticmethod
    def insert_tweet(tweet, ids):
        """insert a tweet into the DB. ids should contain all ids that are in the database."""
        tweet_id = None

        try:
            if tweet is not None:
                if tweet.id is not None:
                    tweet_id = tweet.id
                    if tweet.id not in ids:

                        tweet_attr = dir(tweet)
                        tweet_attr = [s for s in tweet_attr if not
                        DatabaseHandler.prog.match(s)]
                        attr_string = ''
                        attr_value_string = ''
                        for i in tweet_attr:
                            if (i == 'entities') or (i == 'location') or (i == 'place') or (i == 'user') or (
                                        i == 'retweeted_status'):
                                attr_string += i + '_id,'
                                attr_value_string += '%s,'
                            elif i == 'quoted_status':
                                continue
                            else:
                                attr_string += i + ","
                                attr_value_string += '%s,'

                        attr_string = attr_string[:-1]
                        attr_value_string = attr_value_string[:-1]

                        insert_tweet_query = """INSERT INTO tweet (""" + attr_string + """)
                                                VALUES(""" + attr_value_string + """)
                                                ON DUPLICATE KEY UPDATE id=id;"""

                        entities = DatabaseHandler.insert_entities(tweet.entities)
                        location = DatabaseHandler.insert_location(tweet.location)
                        place = DatabaseHandler.insert_place(tweet.place)
                        # changed: no longer crawl referred tweets
                        if tweet.quoted_status is not None:
                            quoted_status = tweet.quoted_status.id
                        else:
                            quoted_status = None
                        if tweet.retweeted_status is not None:
                            retweeted_status = tweet.retweeted_status.id
                        else:
                            retweeted_status = None
                        # quoted_status = DatabaseHandler.insert_tweet(tweet.quoted_status, ids)
                        # retweeted_status = DatabaseHandler.insert_tweet(tweet.retweeted_status, ids)
                        user = DatabaseHandler.insert_user(tweet.user)

                        cur = DatabaseHandler.db.cursor()
                        cur.execute(insert_tweet_query,
                                    (tweet.created_at,
                                     tweet.current_user_retweet,
                                     entities,
                                     tweet.fake,
                                     tweet.favorite_count,
                                     tweet.favorited,
                                     tweet.filter_level,
                                     tweet.id,
                                     tweet.in_reply_to_screen_name,
                                     tweet.in_reply_to_status_id,
                                     tweet.in_reply_to_user_id,
                                     tweet.is_quote_status,
                                     tweet.lang,
                                     location,
                                     place,
                                     tweet.possibly_sensitive,
                                     quoted_status,
                                     tweet.retweet_count,
                                     tweet.retweeted,
                                     retweeted_status,
                                     json.dumps(tweet.scopes),
                                     tweet.source,
                                     tweet.text,
                                     tweet.truncated,
                                     user,
                                     tweet.withheld_copyright,
                                     tweet.withheld_in_countries,
                                     tweet.withheld_scope))
                        DatabaseHandler.db.commit()
                        cur.close()

        except Exception as e:
            print(str(e) + ' tweet id: ' + str(tweet.id))

        return tweet_id

    @staticmethod
    def insert_user(user):
        """insert a user into the DB"""

        user_attr = dir(user)
        user_attr = [s for s in user_attr if not DatabaseHandler.prog.match(s)]
        user_attr_string = ''
        user_real_attr_string = ''
        on_dup_string = ''
        for i in user_attr:
            user_attr_string += i + ','
            user_real_attr_string += '%s,'
            on_dup_string += i + '=%s,'

        user_attr_string = user_attr_string[:-1]
        user_real_attr_string = user_real_attr_string[:-1]
        on_dup_string = on_dup_string[:-1]

        user_url = DatabaseHandler.insert_url(user.url)

        insert_user_query = """INSERT INTO user (""" + user_attr_string + """)
                                VALUES(""" + user_real_attr_string + """) ON DUPLICATE KEY UPDATE """ + on_dup_string + """;"""
        cur = DatabaseHandler.db.cursor()
        cur.execute(insert_user_query,
                    (user.contributors_enabled,
                     user.created_at,
                     user.default_profile,
                     user.default_profile_image,
                     user.description,
                     user.favourites_count,
                     user.followers_count,
                     user.friends_count,
                     user.geo_enabled,
                     user.has_extended_profile,
                     user.id,
                     user.is_translator,
                     user.lang,
                     user.listed_count,
                     user.location,
                     user.name,
                     user.notifications,
                     user.profile_background_color,
                     user.profile_background_image_url,
                     user.profile_background_tile,
                     user.profile_banner_url,
                     user.profile_image_url,
                     user.profile_link_color,
                     user.profile_sidebar_border_color,
                     user.profile_sidebar_fill_color,
                     user.profile_text_color,
                     user.profile_use_background_image,
                     user.protected,
                     user.screen_name,
                     user.show_all_inline_media,
                     user.statuses_count,
                     user.time_zone,
                     user.translator_type,
                     user_url,
                     user.utc_offset,
                     user.verified,
                     user.withheld_in_countries,
                     user.withheld_scope,
                     user.contributors_enabled,
                     user.created_at,
                     user.default_profile,
                     user.default_profile_image,
                     user.description,
                     user.favourites_count,
                     user.followers_count,
                     user.friends_count,
                     user.geo_enabled,
                     user.has_extended_profile,
                     user.id,
                     user.is_translator,
                     user.lang,
                     user.listed_count,
                     user.location,
                     user.name,
                     user.notifications,
                     user.profile_background_color,
                     user.profile_background_image_url,
                     user.profile_background_tile,
                     user.profile_banner_url,
                     user.profile_image_url,
                     user.profile_link_color,
                     user.profile_sidebar_border_color,
                     user.profile_sidebar_fill_color,
                     user.profile_text_color,
                     user.profile_use_background_image,
                     user.protected,
                     user.screen_name,
                     user.show_all_inline_media,
                     user.statuses_count,
                     user.time_zone,
                     user.translator_type,
                     user_url,
                     user.utc_offset,
                     user.verified,
                     user.withheld_in_countries,
                     user.withheld_scope))
        DatabaseHandler.db.commit()
        cur.close()
        return user.id

    @staticmethod
    def insert_url(url):
        cur = DatabaseHandler.db.cursor()
        if url is not None:
            cur.execute("""INSERT INTO url (url,display_url,expanded_url)
                            VALUES(%s,%s,%s) ON DUPLICATE KEY UPDATE url=url;""",
                        (url.url,
                         url.display_url,
                         url.expanded_url))
            DatabaseHandler.db.commit()
            return url.url
        else:
            return None

    @staticmethod
    def insert_location(location):
        """insert a location into the DB"""

        location_id = None
        if location is not None:
            insert_location_query = """INSERT INTO location (latitude,longitude) VALUES(%s,%s);"""
            cur = DatabaseHandler.db.cursor()
            cur.execute(insert_location_query, (location.latitude, location.longitude))
            DatabaseHandler.db.commit()

            location_id = cur.lastrowid
            cur.close()

        return location_id

    @staticmethod
    def insert_place(place):
        """insert a place into the DB"""
        place_id = None

        if place is not None:

            place_attr = dir(place)
            place_attr = [s for s in place_attr if not DatabaseHandler.prog.match(s)]
            place_attr_string = ''
            place_real_attr_string = ''
            for i in place_attr:
                place_attr_string += i + ","
                place_real_attr_string += "%s,"

            place_attr_string = place_attr_string[:-1]
            place_real_attr_string = place_real_attr_string[:-1]

            insert_place_query = """INSERT INTO place (""" + place_attr_string + """) VALUES(""" + place_real_attr_string + """)
                                     ON DUPLICATE KEY UPDATE id=id;"""
            cur = DatabaseHandler.db.cursor()
            cur.execute(insert_place_query,
                        (json.dumps(place.attributes),
                         str(place.bounding_box_coordinates),
                         place.bounding_box_type,
                         place.country,
                         place.country_code,
                         place.full_name,
                         place.id,
                         place.name,
                         place.place_type,
                         place.url))
            DatabaseHandler.db.commit()
            cur.close()
            place_id = place.id
        return place_id

    @staticmethod
    def insert_entities(entity):
        """insert an entity into the DB"""
        entity_id = None
        if entity is not None:
            if (entity.hashtags is not None
                or entity.symbols is not None
                or entity.media is not None
                or entity.urls is not None
                or entity.user_mentions is not None
                or entity.extended_entity is not None):

                cur_tmp = DatabaseHandler.db.cursor()
                create_entity = """INSERT INTO entity (id) VALUES (%s);"""
                cur_tmp.execute(create_entity, 0)
                DatabaseHandler.db.commit()

                entity_id = cur_tmp.lastrowid

                if entity.hashtags is not None or entity.hashtags:
                    for i in entity.hashtags:
                        cur_tmp.execute("""INSERT INTO hashtag (hashtag)
                                        VALUES (%s) ON DUPLICATE KEY UPDATE hashtag=hashtag;""", i)
                        DatabaseHandler.db.commit()

                        cur_tmp.execute("""INSERT INTO entity_hashtag (hashtag,entity_id) VALUES(%s,%s);""",
                                        (i, entity_id))
                        DatabaseHandler.db.commit()

                if entity.symbols is not None or entity.symbols:
                    for i in entity.symbols:
                        cur_tmp.execute("""INSERT INTO symbol (text) VALUES (%s) ON DUPLICATE KEY UPDATE text=text;""",
                                        i)
                        DatabaseHandler.db.commit()

                        cur_tmp.execute("""INSERT INTO entity_symbol (symbol_text,entity_id) VALUES(%s,%s);""",
                                        (i, entity_id))
                        DatabaseHandler.db.commit()

                if entity.urls is not None or entity.urls:
                    for i in entity.urls:
                        cur_tmp.execute("""INSERT INTO url (url,display_url,expanded_url)
                                        VALUES(%s,%s,%s) ON DUPLICATE KEY UPDATE url=url;""",
                                        (i.url,
                                         i.display_url,
                                         i.expanded_url))
                        DatabaseHandler.db.commit()

                        cur_tmp.execute("""INSERT INTO entity_url (url_url,entity_id) VALUES(%s,%s);""",
                                        (i.url, entity_id))
                        DatabaseHandler.db.commit()

                if entity.media is not None or entity.media:
                    media = entity.media
                    media_attr = dir(Media())
                    media_attr = [s for s in media_attr if not DatabaseHandler.prog.match(s)]

                    media_attr_string = ''
                    for i in media_attr:
                        media_attr_string += i + ","
                    media_attr_string = media_attr_string[:-1]

                    for i in media:
                        media_real_attr_string = ''
                        for j in media_attr:
                            media_real_attr_string += "%s,"
                        media_real_attr_string = media_real_attr_string[:-1]

                        media_insert_query = """INSERT INTO media (""" + media_attr_string + """)
                                VALUES(""" + media_real_attr_string + """) ON DUPLICATE KEY UPDATE id=id;"""
                        cur_tmp.execute(media_insert_query,
                                        (i.display_url,
                                         i.expanded_url,
                                         i.id,
                                         i.media_url,
                                         json.dumps(i.sizes),
                                         i.source_status_id,
                                         i.type,
                                         i.url))
                        DatabaseHandler.db.commit()

                        cur_tmp.execute("""INSERT INTO entity_media (media_id,entity_id)
                                        VALUES(%s,%s);""", (i.id, entity_id))
                        DatabaseHandler.db.commit()

                if entity.user_mentions is not None or entity.user_mentions:
                    for i in entity.user_mentions:
                        user_mentions_insert_query = """INSERT INTO user_mentions (name,screen_name,id)
                                        VALUES (%s,%s,%s) ON DUPLICATE KEY UPDATE id=id;"""
                        cur_tmp.execute(user_mentions_insert_query,
                                        (i.name,
                                         i.screen_name,
                                         i.id))
                        DatabaseHandler.db.commit()

                        cur_tmp.execute("""INSERT INTO entity_user_mentions (user_mentions_id,entity_id)
                                        VALUES(%s,%s);""", (i.id, entity_id))
                        DatabaseHandler.db.commit()

                if entity.extended_entity is not None or entity.extended_entity:
                    ee = entity.extended_entity
                    ee_attr = dir(ExtendedEntity())
                    ee_attr = [s for s in ee_attr if not DatabaseHandler.prog.match(s)]

                    ee_attr_string = ''
                    for i in ee_attr:
                        ee_attr_string += i + ","

                    ee_attr_string = ee_attr_string[:-1]

                    ee_real_attr_string = ''
                    for i in ee:
                        for j in ee_attr:
                            ee_real_attr_string = "%s,"
                        ee_real_attr_string = ee_real_attr_string[:-1]

                        extended_entity_insert_query = """INSERT INTO extended_entity(""" + ee_attr_string + """)
                                    VALUES(""" + ee_real_attr_string + """) ON DUPLICATE KEY UPDATE id=id;"""
                        cur_tmp.execute(extended_entity_insert_query,
                                        (i.display_url,
                                         i.duration_millis,
                                         i.expanded_url,
                                         i.id,
                                         i.media_url,
                                         i.sizes,
                                         i.type,
                                         i.variants,
                                         i.video_info,
                                         i.url))
                        DatabaseHandler.db.commit()

                        cur_tmp.execute(
                            """INSERT INTO entity_extended_entity (extended_entity_id,entity_id)
                            VALUES(%s,%s);""", (i.id, entity_id))
                        DatabaseHandler.db.commit()

                cur_tmp.close()
        return entity_id

    @staticmethod
    def expand_urls_in_db():
        """expands all urls in the database that where not expanded before and stores it in the respective column"""

        # expand urls that have not been expanded by the Twitter API
        query = "SELECT url FROM url " \
                "WHERE expanded_url is null;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        urls = [item['url'] for item in cur.fetchall()]
        cur.close()
        print("Expand urls...")
        for u in urls:
            manual_expanded_url = UrlUtils.expand_url(u)
            if manual_expanded_url is not None:
                print(u + " expanded to " + manual_expanded_url)
                insert_query = "UPDATE url SET expanded_url = %s WHERE url = %s;"
                cur = DatabaseHandler.db.cursor()
                cur.execute(insert_query, (manual_expanded_url, u))
                DatabaseHandler.db.commit()
                cur.close()

        # expand expanded urls another time to avoid shortend urls as an expansion
        query = "SELECT expanded_url FROM url " \
                "WHERE manual_expanded_url is null;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        expanded_urls = [item['expanded_url'] for item in cur.fetchall()]
        cur.close()
        print("...expand expanded urls...")
        for u in expanded_urls:
            manual_expanded_url = UrlUtils.expand_url(u)
            if manual_expanded_url is not None:
                print(u + " expanded to " + manual_expanded_url)
                insert_query = "UPDATE url SET manual_expanded_url = %s WHERE expanded_url = %s;"
                cur = DatabaseHandler.db.cursor()
                cur.execute(insert_query, (manual_expanded_url, u))
                DatabaseHandler.db.commit()
                cur.close()

    @staticmethod
    def remove_quotes():
        """collects all quotes. Removes them from the database"""
        print("fetch user ids...")
        user_ids = DatabaseHandler.get_user_ids()

        print("fetch quote ids...")
        sel_quote = "SELECT user_id, quoted_status_id FROM tweet WHERE quoted_status_id is not null"
        cur = DatabaseHandler.db.cursor()
        cur.execute(sel_quote)
        DatabaseHandler.db.commit()
        res = cur.fetchall()
        quoted_ids = list()
        for r in res:
            if r['user_id'] not in user_ids:
                quoted_ids.append(r['quoted_status_id'])
        cur.close()

        # delete_quotes = "DELETE FROM tweet where key_id = %s"
        # cur = DatabaseHandler.db.cursor()
        # cur.executemany(delete_quotes, quotes_ids)
        # DatabaseHandler.db.commit()
        # cur.close()

        print("remove quotes...")
        delete_quoted = "DELETE FROM tweet where id = %s;"
        cur = DatabaseHandler.db.cursor()
        cur.executemany(delete_quoted, quoted_ids)
        DatabaseHandler.db.commit()
        cur.close()

    @staticmethod
    def remove_retweets():
        """collects all retweets. Removes them from the database"""
        print("fetch user ids...")
        user_ids = DatabaseHandler.get_user_ids()

        print("fetch retweet ids...")
        sel_quote = "SELECT user_id, retweeted_status_id FROM tweet WHERE retweeted_status_id is not null;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(sel_quote)
        DatabaseHandler.db.commit()
        res = cur.fetchall()

        retweeted_ids = list()
        for r in res:
            if r['user_id'] not in user_ids:
                retweeted_ids.append(r['retweeted_status_id'])
        cur.close()

        # delete_quotes = "DELETE FROM tweet where key_id = %s"
        # cur = DatabaseHandler.db.cursor()
        # cur.executemany(delete_quotes, quotes_ids)
        # DatabaseHandler.db.commit()
        # cur.close()

        print("remove retweets...")
        delete = "DELETE FROM tweet where id = %s;"
        cur = DatabaseHandler.db.cursor()
        cur.executemany(delete, retweeted_ids)
        DatabaseHandler.db.commit()
        cur.close()

    @staticmethod
    def clear_column(tablename, columnname):
        clear = "UPDATE " + tablename + " SET " + columnname + " = null;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(clear)
        DatabaseHandler.db.commit()
        cur.close()
        print("Column {} in table {} cleared.".format(columnname, tablename))

    @staticmethod
    def preprocessing_values_missing(testset):
        columns = ["tokenized_text","sent_tokenized_text","pos_tags","additional_preprocessed_text",
                   "ascii_emojis","unicode_emojis", "contains_spelling_mistake","nr_of_sentiment_words",
                   "sentiment_score", "subjectivity_score", "is_ww_trending_topic"]

        for column in columns:
            if testset:
                select = "SELECT count(*) as res FROM tweet, testset WHERE tweet.id = testset.tweet_id AND "+column+" is null;"
            else:
                select = "SELECT count(*) as res FROM tweet WHERE "+column+" is null;"
            cur = DatabaseHandler.db.cursor()
            cur.execute(select)
            DatabaseHandler.db.commit()
            res = cur.fetchone()["res"]
            cur.close()
            print("Missing values in column {}: {}".format(column, res))

    @staticmethod
    def insert_user_crawled_original(insert_collection):
        """
        fills a table with the selected users and their respective source
        :param insert_collection: 
        :return: 
        """
        insert = """INSERT INTO user_crawled_original (name,fake,source) VALUES(%s,%s,%s)"""
        cur = DatabaseHandler.db.cursor()
        cur.executemany(insert, insert_collection)
        DatabaseHandler.db.commit()
        cur.close()

    @staticmethod
    def get_time_distribution():
        local_to_remove = ["oswegocotoday",
         "hannibalcourier",
         "bgdailynews",
         "ocalapost",
         "hibbdailytrib",
         "borgernews",
         "apalachtimes",
         "enterprisepub",
         "journalreview",
         "detroitnews",
         "fbheraldnews",
         "bostonirishrptr",
         "dailyherald",
         "dailyrepublic",
         "ospreyobserver"]
        remove_local = " AND ".join(["name != \"" + src + "\"" for src in local_to_remove])

        source_sel_query = " OR ".join(["source = \"" + src + "\"" for src in DatabaseHandler.source_sel])

        user_selection = "(SELECT DISTINCT name FROM " + DatabaseHandler.ACCOUNTS_JOIN_TABLE + " " \
                                                                               "WHERE (" + source_sel_query + ") " \
                                                                               "AND ("+remove_local+")) "

        print("Load dataset...")

        query = "SELECT count(tweet.fake) as count, tweet.fake, YEAR(tweet.created_at) as year, MONTH(tweet.created_at) as month " \
                                            "FROM user, tweet " \
                                            "WHERE lcase(user.screen_name) IN "+user_selection+" " \
                                           "AND tweet.user_id = user.id " \
                                           "AND tweet.created_at between '2015-01-01 00:00:00' and now() " \
                                           "AND tweet.lang = 'en' " \
                + DatabaseHandler.WHERE + " " \
                "Group by year, month, tweet.fake"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        counts = cur.fetchall()
        cur.close()

        return counts

    @staticmethod
    def remove_accounts_from_crawled():
        accounts_to_remove = ["oswegocotoday",
        "hannibalcourier",
        "bgdailynews",
        "ocalapost",
        "hibbdailytrib",
        "borgernews",
        "apalachtimes",
        "enterprisepub",
        "journalreview",
        "detroitnews",
        "fbheraldnews",
        "bostonirishrptr",
        "dailyherald",
        "dailyrepublic",
        "ospreyobserver"]

        for account in accounts_to_remove:
            query = "DELETE FROM " + DatabaseHandler.ACCOUNTS_JOIN_TABLE + " WHERE name = %s"
            cur = DatabaseHandler.db.cursor()
            cur.execute(query, account)
            DatabaseHandler.db.commit()
            cur.close()

    @staticmethod
    def get_user_attribute_names():
        user_cols = DatabaseHandler.get_column_names("user")
        return ["user__"+col for col in user_cols]

    @staticmethod
    def get_user_features_df(testset):
        """
        returns all attributes of a user, except the user id
        :param user_id: 
        :param testset: if true, loads the user for the testset
        :return: 
        """
        user_cols = DatabaseHandler.get_column_names("user")
        aliases = ', '.join(generate_alias_select_sting(user_cols, "user"))

        if testset:
            query = "SELECT "+aliases+ " FROM user WHERE lcase(user.screen_name) IN (SELECT screen_name FROM testset);"
        else:
            query = "SELECT "+aliases+ " FROM user WHERE lcase(user.screen_name) IN "+DatabaseHandler.USER_SELECTION+";"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        user = cur.fetchall()
        cur.close()

        return pd.DataFrame(user)

    @staticmethod
    def insert_ids_testset():
        from Utility.CSVUtils import read_csv
        tweets = read_csv('../data/testdata/politifact_true.csv')
        tweets = [(t[0],t[1].lower(),'true',0) for t in tweets]

        insert = """INSERT INTO testset (tweet_id,screen_name,source,fake) VALUES(%s,%s,%s,%s)"""
        cur = DatabaseHandler.db.cursor()
        cur.executemany(insert, tweets)
        DatabaseHandler.db.commit()

    @staticmethod
    def get_users_testset():
        query = "SELECT u.* FROM user u, testset t " \
                "WHERE lcase(u.screen_name) = t.screen_name;"

        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        list = cur.fetchall()
        cur.close()
        return list

    @staticmethod
    def get_feature_by_tweet_id(feature_name, tweet_id):
        """
        returns feature value by tweet_id
        :param feature_name: 
        :param tweet_id: 
        :return: 
        """
        query = "SELECT "+feature_name+" FROM tweet WHERE id = %s;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query, tweet_id)
        DatabaseHandler.db.commit()
        res = cur.fetchone()[feature_name]
        # res = cur.fetchall()
        cur.close()
        if res is None:
            raise Exception("{} of tweet {} is None!".format(feature_name,tweet_id))
        return res

    @staticmethod
    def remove_ids_not_in_train_or_test():
        """
        removes ids from DB that are neither in train nor in test
        :return: 
        """
        query = "SELECT id from tweet;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        ids_in_db = cur.fetchall()
        ids_in_db = [i['id'] for i in ids_in_db]

        ids_to_keep = list()
        train_ids = get_dataset('nb')['tweet__id'].tolist()
        test_ids = get_testset('nb')['tweet__id'].tolist()

        ids_to_keep.extend(train_ids)
        ids_to_keep.extend(test_ids)
        ids_to_keep = set(ids_to_keep)

        ids_to_remove = [id for id in ids_in_db if id not in ids_to_keep]
        print("Keeps {} ids (Removes {})".format((len(ids_in_db)-len(ids_to_remove)), len(ids_to_remove)))

        delete_quoted = "DELETE FROM tweet where id = %s;"
        cur = DatabaseHandler.db.cursor()
        cur.executemany(delete_quoted, ids_to_remove)
        DatabaseHandler.db.commit()
        cur.close()


