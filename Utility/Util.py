import os
import re

import time

from pandas.types.missing import array_equivalent

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def get_mysql_date(date_string):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(date_string, '%a %b %d %H:%M:%S +0000 %Y'))


def generate_alias_select_sting(columns, table_name):
    """creates a string out of column names"""
    return [table_name + "." + col + " AS " +table_name + "__" + col for col in columns]

def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if array_equivalent(ia, ja):
                    dups.append(cs[i])
                    break

    return dups


def write_list_to_file(tweets, filename):
    """writes a list to a file"""
    with open(filename, 'w') as file:
        for tweet in tweets:
            file.write("%s\n" % tweet)


def str_to_bool(string):
    if string == "True" or string == "1" or string == 1 or string == True:
        return True
    else:
        return False


def chunk_list(list_to_split, number_of_splits):
    """splits a list into an arbitrary number of lists"""
    avg = len(list_to_split) / float(number_of_splits)
    out = []
    last = 0.0

    while last < len(list_to_split):
        out.append(list_to_split[int(last):int(last + avg)])
        last += avg

    return out


def chunk_list_max_size(list_to_split, max_size):
    """
    splits a list into n parts with at most max_size elements
    :param list_to_split: 
    :param max_size: 
    :return: 
    """

    out = []
    curr = []
    count = 0
    for element in list_to_split:
        if count == max_size:
            count = 0
            out.append(curr)
            curr = []

            curr.append(element)
            count += 1
        else:
            curr.append(element)
            count += 1

    out.append(curr)
    return out

def list_diff(list1, list2):
    return [i for i in list1 if i not in list2]

def get_root_directory():
    curr = os.getcwd()
    folders = curr.split("\\")
    count = 0
    for i in reversed(folders):
        if i == 'MasterThesisTwitter':
            break
        count += 1
    if count == 0:
        return '/'.join(folders)
    else:
        return '/'.join(folders[:-count])






