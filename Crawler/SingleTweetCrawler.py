import tweepy
from tweepy import OAuthHandler

from Database.DatabaseHandler import DatabaseHandler
from Utility.JsonUtils import get_tweet_from_json

consumer_key = '*****'
consumer_secret = '*****'
access_token = '*****'
access_secret = '*****'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)
status = api.get_status(123456789)
print(status._json)
t = get_tweet_from_json(status._json, False)
ids = DatabaseHandler.get_all_tweet_ids()
DatabaseHandler.insert_tweet(t, ids)
print(t)