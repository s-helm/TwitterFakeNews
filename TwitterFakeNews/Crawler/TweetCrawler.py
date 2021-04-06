import csv
import time

from tweepy import TweepError

from Database.DatabaseHandler import DatabaseHandler
from Utility.AccountFilesHandler import get_accounts
from Utility.JsonUtils import get_tweet_from_json


def writeToCSV(tweeds, filename):
    # write the csv ab: writing to the end, binary
    with open('tweets/%s_tweets.csv' % filename, 'ab') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "text"])
        writer.writerows(tweeds)


# returns a single tweet
def get_tweet(api, tweet_id):
    return api.get_status(tweet_id)


def load_new_tweets(api, fake):
    accounts = get_accounts(fake)

    for acc in accounts:
        load_tweets_of_user_into_database(api, acc, fake)


def load_testset_accounts(api, fake):
    """
    loads the tweets of the accounts for the testset
    :param api: 
    :param fake: 
    :return: 
    """
    from Utility.CSVUtils import read_csv
    accs = list()
    if fake:
        pf_false = read_csv('../data/politifact_false.csv')
        pf_pants_on_fire = read_csv('../data/politifact_pants_on_fire.csv')
        accs.extend([t[1] for t in pf_false])
        accs.extend([t[1] for t in pf_pants_on_fire])
    else:
        pf_true = read_csv('../data/politifact_true.csv')
        accs.extend([t[1] for t in pf_true])
    accs = list(set(accs))
    for acc in accs:
        start_time = time.time()
        load_tweets_of_user_into_database(api, acc, fake)
        time_diff = time.time() - start_time
        if time_diff < 62:
            print("Sleep for {} seconds".format(62-time_diff))
            time.sleep(62-time_diff)


def load_tweets_of_user_into_database(api, screen_name, fake):
    """inserts the tweets of the users timeline into the database"""
    print("Crawl tweets of: \"" + screen_name + "\"...")
    tweets_json = get_tweets_of_user(api, screen_name)

    print("..fetch tweet ids...")
    ids = DatabaseHandler.get_all_tweet_ids()

    print("...insert into db: \"" + screen_name + "\"...")
    for t in tweets_json:
        tweet = get_tweet_from_json(t, fake)

        DatabaseHandler.insert_tweet(tweet, ids)

    DatabaseHandler.insert_user_crawled(screen_name)


def insert_testset_tweets(api, fake):
    """
    inserts the testset tweets that belong to the respective class
    :param api: 
    :param fake: 
    :return: 
    """
    from Utility.CSVUtils import read_csv
    tweet_ids = list()
    if fake:
        pf_false = read_csv('../data/politifact_false.csv')
        pf_pants_on_fire = read_csv('../data/politifact_pants_on_fire.csv')
        tweet_ids.extend([t[0] for t in pf_false])
        tweet_ids.extend([t[0] for t in pf_pants_on_fire])
    else:
        pf_true = read_csv('../data/politifact_true.csv')
        tweet_ids.extend([t[0] for t in pf_true])

    for tweet_id in tweet_ids:
        insert_tweet(api, tweet_id, fake)


def insert_tweet(api, tweet_id, fake):
    """
    inserts a single tweet into the database
    :param api: 
    :param tweet_id: 
    :param fake: 
    :return: 
    """
    ids = DatabaseHandler.get_all_tweet_ids()

    tweet_json = get_tweet(api, tweet_id)
    print("Insert tweet {}...".format(tweet_id))
    tweet = get_tweet_from_json(tweet_json, fake)

    DatabaseHandler.insert_tweet(tweet, ids)


# get the most recent tweeds (only 3200 with this method)
def get_tweets_of_user(api, screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    try:
        # make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, include_rts=True)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        # keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:
            # print
            # "getting tweets before %s" % (oldest)

            # all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

            # save most recent tweets
            alltweets.extend(new_tweets)

            # update the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1

            # print("...%s tweets downloaded so far" % (len(alltweets)))

        # transform the tweepy tweets into a 2D array that will populate the csv
        # outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]

        # return outtweets
        # print(alltweets)
        print("...tweets download of \""+screen_name+"\" completed.")
    except TweepError as te:
        print("TweepError: " + str(te))
    except IndexError as ie:
        print("IndexError: " + str(ie) + "(user does not have any tweets)")
    return alltweets
