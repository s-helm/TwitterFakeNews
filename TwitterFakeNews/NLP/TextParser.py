import re
import unicodedata


class TextParser:


    @staticmethod
    def normalize(text):
        """converts the text into Normalized From C (NFC) text"""
        # tmp = str(text, "utf-8")
        unicode_text = unicodedata.normalize('NFC', text)
        return unicode_text

    @staticmethod
    def find_all_urls(text):
        """searches the text for urls and returns them"""
        search_str = '(?P<url>https?://[^\s]+)'
        result = re.findall(search_str, text)
        return result

    @staticmethod
    def find_all_user_mentions(text):
        """searches the text for user mentions and returns them"""
        SearchStr = '\B@\w*[a-zA-Z]+\w*'
        Result = re.findall(SearchStr, text)
        return Result

    @staticmethod
    def find_all_hashtags(text):
        """searches the text for hashtags and returns them"""
        SearchStr = '\B#\w*[a-zA-Z]+\w*'
        Result = re.findall(SearchStr, text)
        return Result

    @staticmethod
    def find_all_upercase_tokens(tokens):
        upper = list()
        for t in tokens:
            if t.isupper():
                upper.append(t)
        return upper

