import subprocess
from datetime import datetime
import time

import schedule as schedule
from pytz import timezone

from threading import Timer

import tweepy
from tweepy import OAuthHandler
from tweepy.streaming import json

from TrendingTopics.CSVReader import get_prepared_woeid_map
from TrendingTopics.Counter import Counter
from TrendingTopics.DatabaseHandler import connect_db, insert_trending_topics, close_db

woeid_map = get_prepared_woeid_map("woeid_map.csv")
# US: 23424977
# Germany: 23424829
# Worldwide: 1
# US
woeid_map[-18000].add(23424977)
# Worldwide
woeid_map[0].add(1)

def get_twitter_api():
    consumer_key = '******'
    consumer_secret = '******'
    access_token = '******'
    access_secret = '******'

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth)
    return api


def get_trending_topics(api, id):
    trends = None
    try:
        trends = api.trends_place(id)
    except Exception as e:
        pass
    return trends


def write_trending_topics_to_json(tt, filename):
    try:
        filename = 'trending_topics/%s.json' % filename
        with open(filename, 'w+') as outfile:

            json.dump(tt, outfile, indent=4)
            return filename
    except BaseException as e:
        print("Error on_data: ", str(e))


def save_trending_topics(key, c):
    try:
        c.increase_count()
        db = connect_db()
        regions = list(woeid_map[key])

        for key in regions:
            # x = get_eastern_time()
            fmt = '%Y-%m-%d %H:%M:%S %Z%z'
            # fmt2 = '%Y_%m_%d_%H_%M'

            trends = get_trending_topics(get_twitter_api(), key)
            if trends is not None:
                insert_trending_topics(db, trends[0])
            # write_trending_topics_to_json(trends, "trending_topics_"+value+"_" + x.strftime(fmt2))
            # text = "Trending topics saved at: " + datetime.now().strftime(fmt) + " UTC " + str(key / 60 / 60)
            # print(text)
            # send_message(text)

    except Exception as e:
        # raise e
        # send_message(e)
        close_db(db)

    if c.get_count() == len(woeid_map):
        c.reset_count()
        text = "Yesterdays trending topics saved at."
        # print(subprocess.check_output(['msg', 'Stefan_H', text]))
        # send_message(text)


def test(key):
    print("test: " + key)


def get_eastern_time():
    tz = timezone('US/Eastern')
    return datetime.now(tz)


def send_message(message):
    subprocess.Popen(["/home/pi/telegram_send_message.sh", message])


c = Counter()
for key in woeid_map:
    h_offset = key / 60 / 60
    offset_d = 1
    h = 23
    if h_offset > offset_d:
        h = 23 - h_offset + offset_d
    elif h_offset < offset_d:
        h = 0 - (0 + h_offset)

    # print(str(int(h)) + ":59" + " key: " + str(key))
    # save_trending_topics()
    schedule.every().day.at(str(int(h)) + ":59").do(save_trending_topics, key, c)
#
# 8:59 key: -28800
# 7:59 key: -25200
# 10:59 key: -36000
# 6:59 key: -21600
# 4:59 key: -14400
# 0:59 key: 0
# 22:59 key: 7200
# 16:59 key: 28800
# 5:59 key: -18000
# 3:59 key: -10800
# 23:59 key: 3600
# 13:59 key: 39600

# schedule.every().day.at("18:08").do(save_trending_topics, 28800, c)
# schedule.every().day.at("22:59").do(save_trending_topics, 7200)
# schedule.every().day.at("23:59").do(save_trending_topics, 3600)
# schedule.every().day.at("0:59").do(save_trending_topics, -0)
# schedule.every().day.at("3:59").do(save_trending_topics, -10800)
# schedule.every().day.at("4:59").do(save_trending_topics, -14400)
# schedule.every().day.at("5:59").do(save_trending_topics, -18000)
# schedule.every().day.at("6:59").do(save_trending_topics, -21600)
# schedule.every().day.at("7:59").do(save_trending_topics, -25200)
# schedule.every().day.at("8:59").do(save_trending_topics, -28800)
# schedule.every().day.at("10:59").do(save_trending_topics, -36000)
# schedule.every().day.at("13:59").do(save_trending_topics, -39600)


# schedule.every().minute.do(save_trending_topics)

while True:
    schedule.run_pending()
    time.sleep(60)  # wait one minute
