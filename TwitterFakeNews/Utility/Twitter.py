import tweepy
from tweepy import OAuthHandler


def get_twitter_api():
    """connects to twitter"""
    consumer_key = '*****'
    consumer_secret = '*****'
    access_token = '*****'
    access_secret = '*****'

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    return tweepy.API(auth)
