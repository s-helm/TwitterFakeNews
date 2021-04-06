import json

import re
from collections import Counter
from functools import lru_cache

from NLP.NLPUtils import NLPUtils
from Utility.Util import get_root_directory


class Emojis:

    @staticmethod
    def remove_ascii_emojis(text, emojis):
        for e in emojis:
            text = re.sub(re.escape(e), '', text)
        return text

    @staticmethod
    def remove_unicode_emojis(text):
        emojis = Emojis.find_unicode_emojis(text)
        for e in emojis:
            e = NLPUtils.unicode_to_character(e)
            text = re.sub(e, '', text)
        return text

    @staticmethod
    def find_ascii_emojis(text, emojis):
        """returns a map with the occurences of ascii emojis"""
        ctr = Counter()
        for e in emojis:
            occurences = [m.start() for m in re.finditer(re.escape(e), text)]
            # occurences = list(NLPUtils.find_all(re.escape(e), text))
            if occurences:
                # text = text.replace(e, ' ')
                ctr[e] = len(occurences)
        return ctr

    @staticmethod
    def find_unicode_emojis(text, category=None):
        """returns a list with all emoticons in the text"""
        list_uni = list()
        list_bi = list()
        list_tri = list()
        list_quad = list()
        list_pent = list()

        from NLP.TextPreprocessor import TextPreprocessor

        text = TextPreprocessor.remove_urls(text)

        pentagram = NLPUtils.find_ngrams(text, 5)
        for pent in pentagram:
            p = [NLPUtils.character_to_unicode(p) for p in pent]
            unicode_pent = ' '.join(p)
            found_pent = Emojis.unicode_in_emojis(unicode_pent, category)
            if found_pent:
                list_pent.append(unicode_pent)
                text = text.replace(''.join(pent), '')

        quadrigrams = NLPUtils.find_ngrams(text, 4)
        for quad in quadrigrams:
            q = [NLPUtils.character_to_unicode(q) for q in quad]
            unicode_quad = ' '.join(q)
            found_quad = Emojis.unicode_in_emojis(unicode_quad, category)
            if found_quad:
                list_quad.append(unicode_quad)
                text = text.replace(''.join(quad), '')

        trigrams = NLPUtils.find_ngrams(text, 3)
        for tri in trigrams:
            t = [NLPUtils.character_to_unicode(t) for t in tri]
            unicode_tri = ' '.join(t)
            found_tri = Emojis.unicode_in_emojis(unicode_tri, category)
            if found_tri:
                list_tri.append(unicode_tri)
                text = text.replace(''.join(tri), '')

        bigrams = NLPUtils.find_ngrams(text, 2)
        for bi in bigrams:
            b = [NLPUtils.character_to_unicode(b) for b in bi]
            unicode_bi = ' '.join(b)
            found_bi = Emojis.unicode_in_emojis(unicode_bi, category);
            if found_bi:
                list_bi.append(unicode_bi)
                text = text.replace(''.join(bi), '')

        unigrams = NLPUtils.find_ngrams(text, 1)
        for uni in unigrams:
            unicode_uni = NLPUtils.character_to_unicode(''.join(uni))
            found_uni = Emojis.unicode_in_emojis(unicode_uni, category)
            if found_uni:
                list_uni.append(unicode_uni)
                text = text.replace(''.join(uni), '')

        found = list()
        found.extend(list_uni)
        found.extend(list_bi)
        found.extend(list_tri)
        found.extend(list_quad)
        found.extend(list_pent)

        return found

    @staticmethod
    def unicode_emoji_in_category(emoji_list, category):
        count = 0
        category = Emojis.get_all_emojis(category)
        for emoji in emoji_list:
            if emoji in category:
                count += 1
        return count

    @staticmethod
    def unicode_in_emojis(unicode, category = None):
        emojis = Emojis.get_all_emojis(category)
        return unicode in emojis

    @staticmethod
    def read_emoji_map():
        """load unicode emoji codes from json"""
        with open(get_root_directory()+'/resources/emoji_map.json') as data_file:
            data = json.load(data_file)
        return data

    @staticmethod
    @lru_cache(maxsize=2)
    def read_ascii_emojis():
        """load ascii emoji codes from json"""
        with open(get_root_directory()+'/resources/ascii_emojis_datagenetics.json') as data_file:
            data = json.load(data_file)
        return data

    @staticmethod
    def read_unicode_emoji_sents_map():
        """load unicode emojis with sentiment from json"""
        with open(get_root_directory()+'/resources/unicode_emoji_sentiments.json') as data_file:
            data = json.load(data_file)
        return data

    @staticmethod
    def read_ascii_emoji_sents_map():
        """load unicode emojis with sentiment from json"""
        with open(get_root_directory()+'/resources/ascii_emoji_sentiments.json') as data_file:
            data = json.load(data_file)
        return data

    @staticmethod
    @lru_cache(maxsize=8)
    def get_all_emojis(category=None):
        """returns a list of all emoji unicode codes"""
        emojis = list()
        data = Emojis.read_emoji_map()
        if category:
            emojis.extend(data[category])
        else:
            emojis.extend(data['medical'])
            emojis.extend(data['mail'])
            emojis.extend(data['dishware'])
            emojis.extend(data['arrow'])
            emojis.extend(data['face-sick'])
            emojis.extend(data['plant-other'])
            emojis.extend(data['sound'])
            emojis.extend(data['office'])
            emojis.extend(data['body'])
            emojis.extend(data['face-neutral'])
            emojis.extend(data['person-role'])
            emojis.extend(data['animal-bug'])
            emojis.extend(data['animal-amphibian'])
            emojis.extend(data['alphanum'])
            emojis.extend(data['food-vegetable'])
            emojis.extend(data['music'])
            emojis.extend(data['food-prepared'])
            emojis.extend(data['food-sweet'])
            emojis.extend(data['cat-face'])
            emojis.extend(data['lock'])
            emojis.extend(data['warning'])
            emojis.extend(data['monkey-face'])
            emojis.extend(data['other-object'])
            emojis.extend(data['other-symbol'])
            emojis.extend(data['transport-water'])
            emojis.extend(data['transport-ground'])
            emojis.extend(data['sky \u0026 weather'])
            emojis.extend(data['animal-mammal'])
            emojis.extend(data['drink'])
            emojis.extend(data['tool'])
            emojis.extend(data['animal-bird'])
            emojis.extend(data['person-sport'])
            emojis.extend(data['place-other'])
            emojis.extend(data['phone'])
            emojis.extend(data['person'])
            emojis.extend(data['person-activity'])
            emojis.extend(data['geometric'])
            emojis.extend(data['place-geographic'])
            emojis.extend(data['clothing'])
            emojis.extend(data['person-gesture'])
            emojis.extend(data['game'])
            emojis.extend(data['flag'])
            emojis.extend(data['musical-instrument'])
            emojis.extend(data['transport-sign'])
            emojis.extend(data['country-flag'])
            emojis.extend(data['animal-marine'])
            emojis.extend(data['keycap'])
            emojis.extend(data['face-role'])
            emojis.extend(data['computer'])
            emojis.extend(data['place-map'])
            emojis.extend(data['writing'])
            emojis.extend(data['plant-flower'])
            emojis.extend(data['hotel'])
            emojis.extend(data['event'])
            emojis.extend(data['face-positive'])
            emojis.extend(data['creature-face'])
            emojis.extend(data['book-paper'])
            emojis.extend(data['animal-reptile'])
            emojis.extend(data['transport-air'])
            emojis.extend(data['food-asian'])
            emojis.extend(data['zodiac'])
            emojis.extend(data['light \u0026 video'])
            # emojis.extend(data['skin-tone'])
            emojis.extend(data['award-medal'])
            emojis.extend(data['religion'])
            emojis.extend(data['place-building'])
            emojis.extend(data['emotion'])
            emojis.extend(data['place-religious'])
            emojis.extend(data['money'])
            emojis.extend(data['face-negative'])
            emojis.extend(data['food-fruit'])
            emojis.extend(data['av-symbol'])
            emojis.extend(data['time'])
            emojis.extend(data['family'])
            emojis.extend(data['sport'])

        emojis = [e.replace('\\', '\\') for e in emojis]
        return emojis

    @staticmethod
    def get_skins():
        """return the skin unicodes"""
        emojis = list()
        data = Emojis.read_emoji_map()
        emojis.extend(data['skin-tone'])
        return emojis

if __name__ == "__main__":
    text = "⏪ #poll ⏩ Media: 'President Trump is a Russian proxy puppet president and Russia HACKED the election?'Do you agree?"
    print(Emojis.remove_unicode_emojis(text))