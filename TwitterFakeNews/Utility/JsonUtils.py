import os
import re

from collections import Counter
from tweepy.streaming import json
from Domain.Entity import Entity
from Domain.ExtendedEntity import ExtendedEntity
from Domain.Location import Location
from Domain.Media import Media
from Domain.Place import Place
from Domain.Tweet import Tweet
from Domain.Url import Url
from Domain.User import User
from Domain.UserMention import UserMention
from Utility.Util import get_mysql_date


def get_tweet_from_json(tweet_json, fake):
    tweet = Tweet()
    t = None
    try:
        t = json.loads(json.dumps(tweet_json._json))
    except AttributeError:
        t = tweet_json

    tweet_attr = dir(tweet)
    prog = re.compile('(__.*__)')

    tweet_attr = [s for s in tweet_attr if not prog.match(s)]

    for attr in tweet_attr:
        try:
            if attr == 'location':
                location = Location()
                setattr(location, 'longitude', t['coordinates']['coordinates'][0])
                setattr(location, 'latitude', t['coordinates']['coordinates'][1])

                if location.longitude is None:
                    location = None
                setattr(tweet, 'location', location)

            elif attr == 'entities':
                continue

            elif attr == 'place':
                continue

            elif attr == 'user':
                user = User()
                user_attr = dir(user)
                user_attr = [s for s in user_attr if not prog.match(s)]
                u = t['user']
                for u_attr in user_attr:
                    try:
                        if u_attr == 'created_at':
                            date = get_mysql_date(u[u_attr])
                            setattr(user, u_attr, date)

                        elif u_attr == 'url':
                            urls = u['entities']['url']['urls']
                            url = Url()
                            for i in urls:
                                if i['url'] == u[u_attr]:
                                    setattr(url, 'url', i['url'])
                                    setattr(url, 'expanded_url', i['expanded_url'])
                                    setattr(url, 'display_url', i['expanded_url'])
                                    setattr(user, u_attr, url)
                            if url.url is None:
                                setattr(url, 'url', u['url'])
                                setattr(user, u_attr, url)

                        else:
                            setattr(user, u_attr, u[u_attr])
                            # print("set " + u_attr + " to " + u[u_attr])
                    except (TypeError, KeyError) as e:
                        # traceback.print_exc()
                        # print("Type or KeyError (User."+u_attr+"): " + str(e))
                        continue
                setattr(tweet, 'user', user)

            elif attr == 'quoted_status' or attr == 'retweeted_status':
                test_tweet = get_tweet_from_json(t[attr], fake)
                setattr(tweet, attr, test_tweet)

            elif attr == 'created_at':
                date = get_mysql_date(t[attr])
                setattr(tweet, attr, date)
            elif attr == 'fake':
                setattr(tweet, attr, fake)
            else:
                setattr(tweet, attr, t[attr])


        except (TypeError, KeyError) as e:
            # print("Type or KeyError: " + str(e))
            continue

    entity = Entity()
    try:
        hashtags = []
        for i in t['entities']['hashtags']:
            hashtags.append(i['text'])
        setattr(entity, 'hashtags', hashtags)
    except KeyError:
        pass

    try:
        symbols = []
        for i in t['entities']['symbols']:
            hashtags.append(i['text'])
        setattr(entity, 'symbols', symbols)
    except KeyError:
        pass

    try:
        tmp_url = Url()
        url_attr = dir(tmp_url)
        url_attr = [s for s in url_attr if not prog.match(s)]

        urls = []
        for i in t['entities']['urls']:
            url = Url()
            for attr in url_attr:
                try:
                    setattr(url, attr, i[attr])
                except KeyError:
                    continue
            urls.append(url)
        setattr(entity, 'urls', urls)
    except KeyError:
        pass

    try:
        tmp_m = Media()
        m_attr = dir(tmp_m)
        m_attr = [s for s in m_attr if not prog.match(s)]

        medias = []
        for i in t['entities']['media']:
            media = Media()
            for attr in m_attr:
                try:
                    setattr(media, attr, i[attr])
                except KeyError:
                    continue

            medias.append(media)
        setattr(entity, 'media', medias)
    except KeyError:
        pass

    try:
        tmp_um = UserMention()
        um_attr = dir(tmp_um)
        um_attr = [s for s in um_attr if not prog.match(s)]

        user_mentions = []
        for i in t['entities']['user_mentions']:
            user_mention = UserMention()
            for attr in um_attr:
                try:
                    setattr(user_mention, attr, i[attr])
                except KeyError:
                    continue
            user_mentions.append(user_mention)
        setattr(entity, 'user_mentions', user_mentions)
    except KeyError:
        pass

    try:
        tmp_ee = ExtendedEntity()
        ee_attr = dir(tmp_ee)
        ee_attr = [s for s in ee_attr if not prog.match(s)]

        extended_entities = []
        for i in t['entities']['extended_entities']:
            extended_entity = ExtendedEntity()
            for attr in ee_attr:
                try:
                    setattr(extended_entity, attr, i[attr])
                except KeyError:
                    continue
            extended_entities.append(extended_entity)
        setattr(entity, 'extended_entities', extended_entities)
    except KeyError:
        pass

    if ((not entity.hashtags
         and not entity.extended_entity
         and not entity.media
         and not entity.symbols
         and not entity.urls
         and not entity.user_mentions)
        or
            (entity.hashtags is None
             and entity.extended_entity is None
             and entity.media is None
             and entity.symbols is None
             and entity.urls is None
             and entity.user_mentions is None)):
        entity = None

    setattr(tweet, 'entities', entity)

    place = Place()
    try:
        p = t['place']
        setattr(place, 'attributes', p['attributes'])
        setattr(place, 'bounding_box_coordinates', p['bounding_box']['coordinates'])
        setattr(place, 'bounding_box_type', p['bounding_box']['type'])
        setattr(place, 'country', p['country'])
        setattr(place, 'country_code', p['country_code'])
        setattr(place, 'full_name', p['full_name'])
        setattr(place, 'id', p['id'])
        setattr(place, 'name', p['name'])
        setattr(place, 'place_type', p['place_type'])
        setattr(place, 'url', p['url'])
    except TypeError:
        place = None
    setattr(tweet, 'place', place)

    if (tweet.id is None):
        tweet = None

    return tweet


# clears a file
def clear_file(filepath):
    open(filepath, 'w').close()


def write_to_json(tweets, filename):
    try:
        filename = '../tweets/%s_tweets.json' % filename
        with open(filename, 'a') as outfile:
            # outfile.write("[")
            for i, tweet in enumerate(tweets):
                json.dump(tweet._json, outfile, indent=4)
                # if i!= len(tweets)-1:
                outfile.write(',')

            # outfile.write("]")
            return filename
    except BaseException as e:
        print("Error on_data: ", str(e))


def append_file(filename, text):
    with open(filename, 'a') as file:
        file.write(text)


def remove_last_char(filename):
    with open(filename, 'rb+') as filehandle:
        filehandle.seek(-1, os.SEEK_END)
        filehandle.truncate()


def read_accounts(filename):
    import json
    # cwd = os.getcwd()  # Get the current working directory (cwd)


    with open('../accounts/' + filename) as json_data:
        data = json.load(json_data)
        return data['accounts']

def read_json(filename):
    with open(filename) as json_data:
        return json.load(json_data)

def json_string_to_counter(x):
    return Counter(json.loads(x))



