import ast
import json
import re
import string
import pandas as pd
from collections import Counter
from functools import lru_cache
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from NLP.TextParser import TextParser
from Utility.Util import get_root_directory


class NLPUtils:

    @staticmethod
    def generate_n_grams(tokens, n):
        """
        generates n grams
        :param tokens: list of tokens
        :param n: uni, bi, tri,... -gram
        :return: list with n grams
        """
        return [' '.join(n) for n in ngrams(tokens, n)]

    @staticmethod
    def get_stopwords():
        """return the list of stopwords"""
        additional = ['’:', '…:', '...:', '‘:', '“:', '”:', '', '.  ...', '. . . .']
        # RT = retweet
        # NLTK stopwod list
        return list(additional) + stopwords.words('english') + list(string.punctuation)

    @staticmethod
    def get_punctuation():
        additional = ['’', '…', '‘', '“', '”']
        return list(string.punctuation) + additional


    @staticmethod
    @lru_cache(maxsize=1024)
    def character_to_unicode(c):
        """converts a character to the code that represents it in unicode"""
        return 'U+{:5X}'.format(ord(c)).replace(' ', '')

    @staticmethod
    def unicode_to_character(uni):
        """converts conicode in format U+XXXX to a character"""
        # try:
        return chr(int(uni.replace('U+','').lower(),16))
        # except ValueError:
        #     uni.replace('U+', '')

    @staticmethod
    def unicode_to_character_pos_tagged(pos_tags):
        return [{'token': NLPUtils.unicode_to_character(tags['token']), 'tag': tags['tag']} if re.match(
                "U\+[a-zA-Z0-9]{2,5}", tags['token']) else {'token': tags['token'], 'tag': tags['tag']} for
            tags in pos_tags]

    @staticmethod
    def find_ngrams(input_list, n):
        """finds ngrams in a list"""
        return zip(*[input_list[i:] for i in range(n)])

    @staticmethod
    def sentence_tokenize(text):
        """tokenizes the text into sentences"""
        return sent_tokenize(text)

    @staticmethod
    def get_slang_abbreviations():
        """returns a dictionary with twitter specific abbreviations"""
        with open(get_root_directory()+'/resources/twitter_slang_abbreviations.json') as data_file:
            return json.load(data_file)

    @staticmethod
    def get_official_abbreviations():
        """returns a dictionary with official abbreviations like 'US'"""
        with open(get_root_directory()+'/resources/official_abbreviations.json') as data_file:
            return json.load(data_file)

    @staticmethod
    @lru_cache(maxsize=None)
    def get_slang_words():
        """returns a dictionary with twitter specific slang words"""
        with open(get_root_directory()+'/resources/twitter_slang_words.json') as data_file:
            return json.load(data_file)

    @staticmethod
    def str_list_to_list(string):
        if pd.isnull(string):
            string = '[]'
        return ast.literal_eval(string)

    @staticmethod
    def count_upper_case_tokens():
        """counts all tokens that are uppercase"""
        from Database.DatabaseHandler import DatabaseHandler
        tweets = DatabaseHandler.get_tweets('tokenized_text', True)
        upper = list()
        for t in tweets:
            tokens = NLPUtils.str_list_to_list(t)
            upper.extend(TextParser.find_all_upercase_tokens(tokens))
        return Counter(upper).most_common()


