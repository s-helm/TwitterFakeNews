import enchant
from nltk.metrics import edit_distance


class SpellChecker(object):
    """
    requires Python 3.5 32-bit
    """
    def __init__(self, dict_name='en_US', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist
        pass

    def correct(self, word, tag = ''):
        """replaces the word with the best suggested match according to the edit distance. 
        Takes a manual set max_dist into account"""
        if tag != '^':
            if self.spell_dict.check(word):
                return word
            suggestions = self.spell_dict.suggest(word)

            best_dist = 1000
            best_sug = ""
            for sug in reversed(suggestions):
                dist = edit_distance(word, sug)
                if dist <= best_dist:
                    best_dist = dist
                    best_sug = sug

            if suggestions and best_dist <= self.max_dist:
                return best_sug
            else:
                return word
            pass
        else:
            return word


if __name__ == "__main__":
    s = SpellChecker()
    text = ""
    print(s.correct(text))