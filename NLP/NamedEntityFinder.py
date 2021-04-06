import json

from nltk import word_tokenize, pos_tag, ne_chunk


def get_noun_phrases(pos_tagged_tokens):
    all_nouns = []
    previous_pos = None
    current_chunk = []
    for (token, pos) in pos_tagged_tokens:
        if pos.startswith('NN'):
            if pos == previous_pos:
                current_chunk.append(token)
            else:
                if current_chunk:
                    all_nouns.append((' '.join(current_chunk), previous_pos))
                current_chunk = [token]
        else:
            if current_chunk:
                all_nouns.append((' '.join(current_chunk), previous_pos))
            current_chunk = []
        previous_pos = pos
    if current_chunk:
        all_nouns.append((' '.join(current_chunk), pos))
    return all_nouns

def get_entities(tree, entity_type):
    for ne in tree.subtrees():
        if ne.label() == entity_type:
            tokens = [t[0] for t in ne.leaves()]
            yield ' '.join(tokens)

if __name__ == '__main__':

    text = "London"
    ex = json.loads("""[{"token":"International","tag":"N"},{"token":":","tag":","},{"token":"Trump","tag":"^"},{"token":"administration","tag":"N"},{"token":"just","tag":"R"},{"token":"15","tag":"$"},{"token":"inflammatory","tag":"A"},{"token":"statements","tag":"N"},{"token":"away","tag":"R"},{"token":"from","tag":"P"},{"token":"isolating","tag":"V"},{"token":"all","tag":"D"},{"token":"nations","tag":"N"},{"token":"not","tag":"R"},{"token":"called","tag":"V"},{"token":"United","tag":"^"},{"token":"States","tag":"N"},{"token":"of","tag":"P"},{"token":"America","tag":"^"}]""")
    tagged_tokens = list()
    for t in ex:
        token = t['token']
        tag = t['tag']
        if tag == '^':
            tag = 'NNP'
        tagged_tokens.append((token,tag))

    # entity = wikipedia.summary(text, sentences = 2)
    #
    tokens = word_tokenize("United States of America is a country.")
    print(tokens)
    tagged_tokens = pos_tag(tokens)
    print(tagged_tokens)
    chunks = ne_chunk(tagged_tokens, binary=True)

    # print("-----")
    # print("Description of {}".format(text))
    # print(entity)
    # print("-----")
    print("Noun phrases in description:")
    for noun in get_noun_phrases(tagged_tokens):
        print(noun[0])  # tuple (noun, pos_tag)
    print("-----")
    print("Named entities in description:")
    # for ne in get_entities(chunks, entity_type='NE'):
    #     summary = wikipedia.summary(ne, sentences=1)
    #     print("{}: {}".format(ne, summary))
