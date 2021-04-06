#!/usr/bin/python3
import pymysql
import time
from Location import Location
from TrendingTopic import TrendingTopic


def connect_db():
    return pymysql.connect(host='localhost', user='root', password='*****', db='trending_topics', charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)


def close_db(db):
    db.close()


def insert_trending_topics(db, tt):
    # tt = json.loads(json.dumps(tt_json._json))

    trends = tt['trends']

    as_of = get_mysql_date(tt['as_of'])
    name = tt['locations'][0]['name']
    woeid = tt['locations'][0]['woeid']
    l = Location()
    l.name = name
    l.woeid = woeid

    loc_id = get_location_id(db, l)
    if loc_id is None:
        loc_id = insert_location(db, l)
    for t in trends:
        hashtag = t['name'][1:]
        tweet_volume = t['tweet_volume']

        tt = TrendingTopic()
        tt.as_of = as_of
        tt.hashtag = hashtag
        tt.tweet_volume = tweet_volume
        insert_trending_topic(db, tt, loc_id)


def insert_location(db, l):
    loc_id = None
    col_query = "SELECT * " \
                "FROM location " \
                "WHERE woeid = %s;"

    cur = db.cursor()
    cur.execute(col_query, l.woeid)
    db.commit()
    cols = cur.fetchall()
    cur.close()

    if (len(cols) < 1):
        cur = db.cursor()
        cur.execute("""INSERT INTO location (name, woeid)
                    VALUES(%s,%s);""",
                    (l.name,
                     l.woeid))
        db.commit()
        loc_id = cur.lastrowid
        cur.close()

    return loc_id


def insert_trending_topic(db, tt, loc_id):
    cur = db.cursor()
    cur.execute("""INSERT INTO trend (as_of, hashtag, tweet_volume, location_id)
                VALUES(%s,%s,%s,%s);""",
                (tt.as_of,
                 tt.hashtag,
                 tt.tweet_volume,
                 loc_id))
    db.commit()
    cur.close()

def get_location_id(db, l):
    cur = db.cursor()
    cur.execute("""SELECT id FROM location WHERE woeid = %s""", l.woeid)
    db.commit()
    res = cur.fetchone()
    if res:
        res = res['id']
    cur.close()
    return res


def get_mysql_date(date_string):
    return time.strftime('%Y-%m-%d %H:%M:%S',
                         time.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ'))
