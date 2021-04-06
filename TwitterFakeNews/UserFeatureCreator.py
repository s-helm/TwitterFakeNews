import json

from Database.DatabaseHandler import DatabaseHandler as db
from FeatureEngineering.FeatureCreator import contains_hashtag, contains_user_mention, contains_url, string_length, \
    get_avg_user_mentions_per_tweet, get_avg_hashtags_per_tweet, get_avg_urls_per_tweet, \
    get_avg_post_time, get_tweets_per_week, get_tweets_per_month, get_tweets_per_day, get_minimum_time_between_tweets, \
    get_maximum_time_between_tweets, get_median_time_between_tweets, get_avg_time_between_tweets, \
    get_nr_of_retweets_per_tweet, get_nr_of_quotes_per_tweet, get_nr_of_replies_per_tweet, is_translator_type, \
    get_user_lang_counts, get_percent_with_url, get_percent_with_hashtag, get_percent_with_user_mention, \
    get_top_level_domain_of_expanded_url, get_top_level_domain_type
from GoogleMapsAPI.LocationFinder import LocationFinder
from Utility.TimeUtils import TimeUtils


def create_user_features():
    """
    creates the user features and stores them in the DB
    :return: 
    """

    print("---insert-physical-locations----")
    LocationFinder.insert_physical_locations()

    # users = db.get_users(all=True)

    users = db.get_users_testset()

    count = 0
    for u in users:
        count += 1
        print("create features of user #"+str(count))

        id = u['id']
        # location
        db.insert_user_feature(id, 'has_location', user__has_location(u['location']), 'BOOLEAN')
        db.insert_user_feature(id, 'has_physical_address', u['physical_location'] is not None, 'BOOLEAN')
        db.insert_user_feature(id, 'has_country', u['country'] is not None, 'BOOLEAN')

        # user description
        db.insert_user_feature(id, 'has_desc', u['description'] is not None and u['description'] != '', 'BOOLEAN')
        db.insert_user_feature(id, 'desc_contains_hashtags', contains_hashtag(u['description']), 'BOOLEAN')
        db.insert_user_feature(id, 'desc_contains_user_mention', contains_user_mention(u['description']), 'BOOLEAN')
        db.insert_user_feature(id, 'desc_contains_url', contains_url(u['description']), 'BOOLEAN')
        db.insert_user_feature(id, 'desc_length', len(u['description']), 'INT')

        #user url
        db.insert_user_feature(id, 'url_length', string_length(u['url']), 'INT')
        db.insert_user_feature(id, 'has_url', u['url'] is not None, 'BOOLEAN')
        db.insert_user_feature(id, 'url_top_level_domain', get_top_level_domain_of_expanded_url(u['url']), 'VARCHAR(30)')

        # followers, favourites, friends, lists
        db.insert_user_feature(id, 'has_list', u['listed_count'] > 0, 'BOOLEAN')
        db.insert_user_feature(id, 'has_favourites', u['favourites_count'] > 0, 'BOOLEAN')
        db.insert_user_feature(id, 'has_friends', u['friends_count'] > 0, 'BOOLEAN')
        db.insert_user_feature(id, 'favourites_per_follower', favourites_per_follower(u), 'FLOAT')
        db.insert_user_feature(id, 'friends_per_favourite', friends_per_favourite(u), 'FLOAT')
        db.insert_user_feature(id, 'friends_per_follower', friends_per_follower(u), 'FLOAT')
        db.insert_user_feature(id, 'is_following_more_than_100', u['friends_count'] >= 100, 'BOOLEAN')
        db.insert_user_feature(id, 'at_least_30_follower', u['followers_count'] >= 30, 'BOOLEAN')

        # components in users tweets
        db.insert_user_feature(id, 'avg_user_mention_per_tweet', get_avg_user_mentions_per_tweet(id), 'FLOAT')
        db.insert_user_feature(id, 'avg_hashtags_per_tweet', get_avg_hashtags_per_tweet(id), 'FLOAT')
        db.insert_user_feature(id, 'avg_urls_per_tweet', get_avg_urls_per_tweet(id), 'FLOAT')

        db.insert_user_feature(id, 'percent_with_url', get_percent_with_url(id), 'FLOAT')
        db.insert_user_feature(id, 'percent_with_hashtag', get_percent_with_hashtag(id), 'FLOAT')
        db.insert_user_feature(id, 'percent_with_user_mention', get_percent_with_user_mention(id), 'FLOAT')

        # tweet times
        # absolute
        # db.insert_user_feature(id, 'created_days_ago', TimeUtils.days_ago(u['created_at']), 'INT')
        # relative
        db.insert_user_feature(id, 'created_days_ago', TimeUtils.days_ago(u['created_at'], relative_to="2017-07-13 11:14"), 'INT')
        db.insert_user_feature(id, 'created_hour_of_day', u['created_at'].hour, 'INT')
        db.insert_user_feature(id, 'avg_post_time', get_avg_post_time(id), 'INT')
        db.insert_user_feature(id, 'tweets_per_day', get_tweets_per_day(id), 'INT')
        db.insert_user_feature(id, 'tweets_per_week', get_tweets_per_week(id), 'INT')
        db.insert_user_feature(id, 'tweets_per_month', get_tweets_per_month(id), 'INT')
        db.insert_user_feature(id, 'min_time_between_tweets', get_minimum_time_between_tweets(id), 'INT')
        db.insert_user_feature(id, 'max_time_between_tweets', get_maximum_time_between_tweets(id), 'INT')
        db.insert_user_feature(id, 'median_time_between_tweets', get_median_time_between_tweets(id), 'INT')
        db.insert_user_feature(id, 'avg_time_between_tweets', get_avg_time_between_tweets(id), 'INT')

        # nr of tweets/retweets/quotes/replies
        db.insert_user_feature(id, 'more_than_50_tweets', u['statuses_count'] > 50, 'BOOLEAN')
        db.insert_user_feature(id, 'nr_of_retweets', db.get_nr_of_retweets_by_user(id), 'INT')
        db.insert_user_feature(id, 'nr_of_retweets_per_tweet', get_nr_of_retweets_per_tweet(id), 'FLOAT')
        db.insert_user_feature(id, 'nr_of_quotes', db.get_nr_of_quotes_by_user(id), 'INT')
        db.insert_user_feature(id, 'nr_of_quotes_per_tweet', get_nr_of_quotes_per_tweet(id), 'FLOAT')
        db.insert_user_feature(id, 'nr_of_replies', db.get_nr_of_replies_by_user(id), 'INT')
        db.insert_user_feature(id, 'nr_of_replies_per_tweet', get_nr_of_replies_per_tweet(id), 'FLOAT')

        # additional user features
        db.insert_user_feature(id, 'has_profile_background_image', u['profile_background_image_url'] is not None, 'BOOLEAN')
        db.insert_user_feature(id, 'is_translator_type_regular', is_translator_type(u['translator_type']), 'BOOLEAN')
        db.insert_user_feature(id, 'is_english', u['lang'] == 'en', 'BOOLEAN')
        db.insert_user_feature(id, 'has_tweets_in_different_lang', get_user_lang_counts(id) > 1, 'BOOLEAN')
        db.insert_user_feature(id, 'tweets_in_different_lang', get_user_lang_counts(id), 'INT')

        # depends on other features
        # db.insert_user_feature(id, 'uses_retweets', u['nr_of_retweets'] > 0, 'BOOLEAN')
        # db.insert_user_feature(id, 'uses_quotes', u['nr_of_quotes'] > 0, 'BOOLEAN')
        # db.insert_user_feature(id, 'uses_replies', u['nr_of_replies'] > 0, 'BOOLEAN')
        # db.insert_user_feature(id, 'has_no_desc_and_loc', not u['has_desc'] and not u['has_physical_address'], 'BOOLEAN')
        # db.insert_user_feature(id, 'url_tld_type', get_top_level_domain_type(u['url_top_level_domain']), 'BOOLEAN')

        ## exlcuded
        ## db.insert_user_feature(id, 'no_desc_loc_following_more_than_100', u['is_following_more_than_100'] and u['has_no_desc_and_loc'], 'BOOLEAN')
        ## db.insert_user_feature(id, 'has_default_profile_after_two_month', user__has_default_profile_after_two_month(u), 'BOOLEAN')


def favourites_per_follower(x):
    """return #favourties/#followers"""
    return x['favourites_count'] / x['followers_count']


def friends_per_follower(x):
    """return #favourties/#followers"""
    return x['friends_count'] / x['followers_count']


def friends_per_favourite(x):
    """return #friends/#favourties, if favourites = 0 returns 0"""
    if x['favourites_count'] == 0:
        return 0
    else:
        return x['friends_count'] / x['favourites_count']


def user__has_default_profile_after_two_month(u):
    return TimeUtils.month_ago(u['created_at']) >=2 and u['default_profile_image']


def user__has_location(loc):
    loc = loc.replace(' ', '')
    return (loc != '') and (loc is not None)


if __name__ == "__main__":
    create_user_features()

