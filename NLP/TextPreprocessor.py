import html
import json
import re
import requests

from nltk import PorterStemmer, WordNetLemmatizer

from NLP.Emoji import Emojis
from NLP.NLPUtils import NLPUtils
#from NLP.SpellChecker import SpellChecker


class TextPreprocessor:
    STEMMER = PorterStemmer()
    LEMMATIZER = WordNetLemmatizer()
    # comment in to use spell checker (requires Python 3.5 32-bit)
    # SPELL_CHECKER = SpellChecker()
    PUNCTUATION = NLPUtils.get_punctuation()
    STOPWORDS = NLPUtils.get_stopwords()
    # comment in to use ascii emojis (the emoji file path might need to be changed to the correct one (folder 'resources'))
    # ASCII_EMOJIS = Emojis.read_ascii_emojis()

    @staticmethod
    def tokenize_tweet(tokenizer, tweet):
        """tokenizes single tweet"""
        text = TextPreprocessor.preprocess_for_tokenize(tweet)
        tokenized = tokenizer.tokenize(text)
        return tokenized

    @staticmethod
    def additional_text_preprocessing_with_pos(pos_dict):
        """performs additional text preprocessing based on the pos tagged text. Needs 32-bit Python, 
        since no 64-bit Pyenchant version is available"""

        tags_to_lemmatize = ['a', 'n', 'v', 'r']

        pos_dict = TextPreprocessor.find_named_entities(pos_dict)
        if pos_dict is None:
            return None, None
        prepro = list()
        contains_spelling_mistake = False
        for t in pos_dict:
            token = t['token']
            tag = t['tag'].lower()
            if token not in TextPreprocessor.PUNCTUATION and tag != ",":

                token = TextPreprocessor.replace_user_mentions(token)
                token = TextPreprocessor.replace_urls(token)
                replaced = [token]
                for i in replaced:

                    i = TextPreprocessor.replace_all_punctuation(i)
                    if i.lower() not in TextPreprocessor.STOPWORDS and i != 'URL' and i!= 'USERMENTION':
                        if i != "" and not re.match('\B#\w*[a-zA-Z]+\w*', i):
                            before = i
                            i = TextPreprocessor.SPELL_CHECKER.correct(i, tag)
                            if i != before:
                                contains_spelling_mistake = True
                        if tag in tags_to_lemmatize:
                            i = TextPreprocessor.lemmatize(i, tag)
                        i = TextPreprocessor.stem(i, tag)
                    # check again, since stemming, lemmatization or spelling correction can produce stopwords
                    # if i.lower() not in TextPreprocessor.STOPWORDS:
                    if i != 'URL' and i!= 'USERMENTION' and i!='':
                        i = i.lower()
                    if re.match(".*[a-zA-Z]'", i):
                        i = i[:-1]
                    prepro.append(i)
        return prepro, contains_spelling_mistake

    @staticmethod
    def preprocess_for_tokenize(text):
        text = TextPreprocessor.unescape_html(text)
        text = TextPreprocessor.replace_all_uppercase(text)
        text = TextPreprocessor.preprocess_blown_up_word(text)
        text = TextPreprocessor.replace_contractions_text(text)
        text = TextPreprocessor.replace_million(text)

        return text

    @staticmethod
    def preprocess_for_sent_tokenize(text, unicode_emojis, ascii_emojis):
        text = TextPreprocessor.unescape_html(text)
        text = TextPreprocessor.remove_urls(text)
        # text = Emojis.remove_ascii_emojis(text, TextPreprocessor.ASCII_EMOJIS)

        ascii_emojis = json.loads(ascii_emojis)
        for e, count in ascii_emojis.items():
            text = re.sub(re.escape(e), '', text)

        unicode_emojis = NLPUtils.str_list_to_list(unicode_emojis)
        tmp_emojis = list()
        for e in unicode_emojis:
            e = e.split(" ")
            tmp_emojis.extend(e)
        for e in tmp_emojis:
            e = NLPUtils.unicode_to_character(e)
            text = re.sub(e, '', text)
        # text = Emojis.remove_unicode_emojis(text)

        return text

    @staticmethod
    def prepocess_pos_tagged_texts(tweet_tokens):
        """gets a list of tokenized texts and preprocesses each of the token lists"""
        return [TextPreprocessor.additional_text_preprocessing_with_pos(json.loads(t)) for t in tweet_tokens]

    @staticmethod
    def replace_official_abbreviations(tokens, abbrevs):
        """replaces official tokens"""
        res = list()
        for token in tokens:
            if token in abbrevs:
                token = abbrevs[token]
            res.append(token)
        return res

    @staticmethod
    def replace_slang_abbreviations(tokens, abbrevs):
        """replaces slang abbreviations"""
        prepro1 = list()
        for token in tokens:
            # t = token.lower().replace('#', '')
            tmp_token = token.lower()
            if tmp_token in abbrevs:
                tmp = abbrevs[tmp_token].split()
                prepro1.extend(tmp)
            else:
                prepro1.append(token)
        return prepro1

    @staticmethod
    def preprocess_blown_up_word(t):
        # look for a character followed by at least one repetition of itself.
        pattern = re.compile(r"(\w)\1+")
        return pattern.sub(TextPreprocessor.repl, t)

    # a function to perform the substitution we need:
    @staticmethod
    def repl(matchObj):
        char = matchObj.group(1)
        return "%s%s" % (char, char)

    @staticmethod
    def replace_html_whitespaces(html):
        """replaces white spaces"""
        # soup = BeautifulSoup(html, "html.parser")
        # text = soup.get_text(" ", strip=True)
        text = html
        # whitespace unicode chars
        text = text.replace('\xa0', ' ')
        text = text.replace('\ufeff', ' ')
        text = ' '.join(text.split())
        return text

    @staticmethod
    def remove_urls(text):
        """removes urls from the text"""
        text = re.sub('(?P<url>https?://[^\s]+)', '', text)
        return text

    @staticmethod
    def replace_urls(text):
        """removes urls from the text"""
        text = re.sub('(?P<url>https?://[^\s]+)', 'URL', text)
        return text

    @staticmethod
    def remove_hashtags(text):
        """removes hashtags from text"""
        text = re.sub('\B#\w*[a-zA-Z]+\w*', '', text)
        return text

    @staticmethod
    def remove_user_mentions(text):
        """removes hashtags from text"""
        text = re.sub('\B@\w*[a-zA-Z]+\w*', '', text)
        return text

    @staticmethod
    def replace_user_mentions(text):
        """removes hashtags from text"""
        text = re.sub('\B@\w*[a-zA-Z]+\w*', 'USERMENTION', text)
        return text

    @staticmethod
    def replace_slang_words(tokens):
        """replaces slang words"""
        abbrevs = NLPUtils.get_slang_abbreviations()
        without_slang = [abbrevs[t] if abbrevs[t] else t for t in tokens]
        return without_slang

    @staticmethod
    def stem(token, tag):
        """stems a token"""
        stemmer = TextPreprocessor.STEMMER
        if tag != '^':
            return stemmer.stem(token)
        else:
            return token

    @staticmethod
    def lemmatize(token, pos_tag):
        """lemmatizes a token based on its POS-tag.
        Allowed values for pos_tag: a(adjective), n(noun), v(verb), r(adverb)"""
        lemmatizer = TextPreprocessor.LEMMATIZER
        return lemmatizer.lemmatize(token, pos_tag)

    @staticmethod
    def remove_rt(token):
        """removes RT, which stands for 'retweet'"""
        if token == 'rt' or token == 'RT':
            return ''
        else:
            return token

    @staticmethod
    def replace_contractions_text(token):
        """short form replacement from http://speakspeak.com/resources/english-grammar-rules/various-grammar-rules/short-forms-contractions
        Some are ambiguous like 'd= had or would
        DONE BY TweetTokenizer!!"""

        token = token.replace("can't", "can not")
        token = token.replace("Can't", "Can not")
        token = token.replace("n't", " not")
        token = token.replace("I'm", "I am")
        token = token.replace("He's", "He is")
        token = token.replace("he's", "he is")
        token = token.replace("She's", "She is")
        token = token.replace("she's", "she is")
        token = token.replace("It's", "It is")
        token = token.replace("it's", "it is")
        token = token.replace("You're", "You are")
        token = token.replace("you're", "you are")
        token = token.replace("We're", "We are")
        token = token.replace("we're", "we are")
        token = token.replace("They're", "They are")
        token = token.replace("they're", "they are")
        token = token.replace("You've", "You have")
        token = token.replace("you've", "you have")
        token = token.replace("I've", "I have")
        token = token.replace("We've", "We have")
        token = token.replace("we've", "we have")
        token = token.replace("They've", "They have")
        token = token.replace("they've", "they have")
        token = token.replace("Let's", "Let us")
        token = token.replace("let's", "let us")
        token = token.replace("Who's", "Who is")
        token = token.replace("who's", "who is")
        token = token.replace("Who'd", "Who would")
        token = token.replace("who'd", "who would")
        token = token.replace("What's", "What is")
        token = token.replace("what's", "what is")
        token = token.replace("How's", "How is")
        token = token.replace("how's", "how is")
        token = token.replace("When's", "When is")
        token = token.replace("when's", "when is")
        # includes there's, where's
        token = token.replace("here's", "here is")
        token = token.replace("Here's", "Here is")
        token = token.replace("There's", "There is")
        token = token.replace("there'd", "there would")
        token = token.replace("There'd", "There would")
        token = token.replace("that's", "that is")
        token = token.replace("That's", "That is")
        # 'll always resolves to ' will
        token = token.replace("'ll", " will")

        # ambiguous would and had
        token = token.replace("I'd", "I would")
        token = token.replace("He'd", "He would")
        token = token.replace("he'd", "he would")
        token = token.replace("She'd", "She would")
        token = token.replace("she'd", "she would")
        token = token.replace("It'd", "It would")
        token = token.replace("it'd", "it would")
        token = token.replace("You'd", "You would")
        token = token.replace("you'd", "you would")
        token = token.replace("We'd", "We would")
        token = token.replace("we'd", "we would")
        token = token.replace("They'd", "They would")
        token = token.replace("they'd", "they would")

        return token

    @staticmethod
    def replace_contractions_token(token):
        """short form replacement from http://speakspeak.com/resources/english-grammar-rules/various-grammar-rules/short-forms-contractions
        Some are ambiguous like 'd= had or would
        DONE BY TweetTokenizer!!"""

        token = TextPreprocessor.replace_short_form_match_part(token, "n't", " not")
        token = TextPreprocessor.replace_short_form_match_whole(token, "I'm", "I am")
        token = TextPreprocessor.replace_short_form_match_whole(token, "He's", "He is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "he's", "he is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "She's", "She is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "she's", "she is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "It's", "It is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "it's", "it is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "You're", "You are")
        token = TextPreprocessor.replace_short_form_match_whole(token, "you're", "you are")
        token = TextPreprocessor.replace_short_form_match_whole(token, "We're", "We are")
        token = TextPreprocessor.replace_short_form_match_whole(token, "we're", "we are")
        token = TextPreprocessor.replace_short_form_match_whole(token, "They're", "They are")
        token = TextPreprocessor.replace_short_form_match_whole(token, "they're", "they are")
        token = TextPreprocessor.replace_short_form_match_whole(token, "You've", "You have")
        token = TextPreprocessor.replace_short_form_match_whole(token, "you've", "you have")
        token = TextPreprocessor.replace_short_form_match_whole(token, "I've", "I have")
        token = TextPreprocessor.replace_short_form_match_whole(token, "We've", "We have")
        token = TextPreprocessor.replace_short_form_match_whole(token, "we've", "we have")
        token = TextPreprocessor.replace_short_form_match_whole(token, "They've", "They have")
        token = TextPreprocessor.replace_short_form_match_whole(token, "they've", "they have")
        token = TextPreprocessor.replace_short_form_match_whole(token, "Let's", "Let us")
        token = TextPreprocessor.replace_short_form_match_whole(token, "let's", "let us")
        token = TextPreprocessor.replace_short_form_match_whole(token, "Who's", "Who is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "who's", "who is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "Who'd", "Who would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "who'd", "who would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "What's", "What is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "what's", "what is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "How's", "How is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "how's", "how is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "When's", "When is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "when's", "when is")
        # includes there's, where's
        token = TextPreprocessor.replace_short_form_match_part(token, "here's", "here is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "Here's", "Here is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "There's", "There is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "there'd", "there would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "There'd", "There would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "that's", "that is")
        token = TextPreprocessor.replace_short_form_match_whole(token, "That's", "That is")
        # 'll always resolves to ' will
        token = TextPreprocessor.replace_short_form_match_part(token, "'ll", " will")

        # ambiguous would and had
        token = TextPreprocessor.replace_short_form_match_whole(token, "I'd", "I would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "He'd", "He would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "he'd", "he would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "She'd", "She would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "she'd", "she would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "It'd", "It would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "it'd", "it would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "You'd", "You would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "you'd", "you would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "We'd", "We would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "we'd", "we would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "They'd", "They would")
        token = TextPreprocessor.replace_short_form_match_whole(token, "they'd", "they would")

        return token

    @staticmethod
    def replace_short_form_match_part(token, str_to_match, replacement):
        if type(token) == list:
            return token
        else:
            if str_to_match in token:
                return replacement.split(" ")
            else:
                return [token]

    @staticmethod
    def replace_short_form_match_whole(token, str_to_match, replacement):
        if type(token) == list:
            return token
        else:
            if str_to_match == token:
                return replacement.split(" ")
            else:
                return [token]

    @staticmethod
    def replace_all_uppercase(text):
        """lowers sequences of at least 5 uppercase characters. Avoid to lower abbreviations like 'US' """
        for match in re.finditer(r'([A-Z]+(!|\.|,)?(.)?){5,}', text):
            text = text.replace(match.group(0), match.group(0).lower())
        return text

    @staticmethod
    def replace_million(text):
        """replaces constructs like '80 M ' with '80 million"""
        for match in re.finditer(r'(£|\$)([0-9]+(\.|,))?[0-9]+(m|M) ', text):
            match_text = match.group(0)
            new_text = match_text.replace('m', ' million').replace('M', ' million')
            text = text.replace(match_text, new_text)
        for match in re.finditer(r'(£|\$)([0-9]+(\.|,))?[0-9]+( |-)(m|M) ', text):
            match_text = match.group(0)
            new_text = match_text.replace('m', 'million').replace('M', 'million')
            text = text.replace(match_text, new_text)
        return text

    @staticmethod
    def replace_all_punctuation(token, percent=0.5):
        """replace tokens whose characters contain at least 'percent' % punctuation"""
        puncts = 0
        for i in range(len(token)):
            if token[i] in TextPreprocessor.PUNCTUATION:
                puncts += 1
            # early stopping if no punctuation after x percent
            if int(len(token) * percent) == i and puncts == 0:
                return token
        if puncts == 0 or len(token) * percent > puncts:
            return token
        else:
            return ""

    @staticmethod
    def unescape_html(text):
        return html.unescape(text)

    @staticmethod
    def find_named_entities(pos_tags):
        """Receives a pos tagged tweet, finds named entities using DBPedia Spotlight 
        and joins the found entities into a single tag"""
        contains_proper_noun = False
        tokens = list()
        for tags in pos_tags:
            if tags['tag'] == '^':
                contains_proper_noun = True

        if contains_proper_noun:
            for tags in pos_tags:
                if len(tags['token']) == 1:
                    tags['token'] = NLPUtils.character_to_unicode(tags['token'])
                tokens.append(tags['token'])
            try:
                text = ' '.join(tokens)
                headers = {
                    'Accept': 'application/json',
                }
                # print(text)
                data = [
                    ('text', text),
                    ('confidence', '0.25'),
                    ('support', '20')
                ]

                r = requests.post('http://model.dbpedia-spotlight.org/en/annotate', headers=headers, data=data,
                                  timeout=10)
                # print(str(r.content.decode()))
                res = r.json()

                entities = list()
                if 'Resources' in res:
                    for i in res['Resources']:
                        # res_str = str(i).replace(',','\n')
                        # print(res_str)

                        if i['@types'] is not None:
                            original = i['@surfaceForm']
                            entity_tmp = i['@URI']
                            entity_tmp = re.sub('.*/', '', entity_tmp)
                            entity_tmp = re.sub('\(.*\)', '', entity_tmp)
                            entity = re.sub('_', ' ', entity_tmp).strip()

                            if entity.lower() in text.lower() and ' ' in entity:
                                entities.append((entity, int(i['@offset'])))
                    # print(entities)
                    new_pos_tags = list()
                    curr_pos = 0
                    tokens_to_omit = 0
                    for tags in pos_tags:
                        # if re.match("U\+[a-zA-Z0-9]{1,5}",tags['token']):
                        #     print(tags['token'])
                        #     tags['token'] = NLPUtils.unicode_to_character(tags['token'])
                        #     print(tags['token'])

                        token = tags['token']
                        for e in entities:
                            curr_dict = dict()
                            if curr_pos == e[1]:
                                tokens_to_omit = len(re.split(' ', e[0]))
                                curr_dict['token'] = e[0]
                                curr_dict['tag'] = '^'
                                new_pos_tags.append(curr_dict)
                        # +1 for whitespace
                        curr_pos += len(token) + 1
                        if tokens_to_omit == 0:
                            new_pos_tags.append(tags)
                        else:
                            tokens_to_omit -= 1

                    # decode unicode sequence
                    new_pos_tags = NLPUtils.unicode_to_character_pos_tagged(new_pos_tags)
                    return new_pos_tags
                # decode uniocde character
                pos_tags = NLPUtils.unicode_to_character_pos_tagged(pos_tags)
            except Exception as e:
                print(e)
                return None

        return pos_tags


    @staticmethod
    def remove_stopwords(tokens):
        return [token for token in tokens if token not in TextPreprocessor.STOPWORDS]
