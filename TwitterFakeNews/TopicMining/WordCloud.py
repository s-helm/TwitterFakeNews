import os
import re

from os import path
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from wordcloud import WordCloud
from TextRepresentation.TextModel import TextModel

d = path.dirname(__file__)

# Read the whole text.

def build_wordcloud(data, topic_nr, total_topics, write_to_file=False):

    # Generate a word cloud image
    # wordcloud = WordCloud().generate(text)

    # Display the generated image:
    # the matplotlib way:
    import matplotlib.pyplot as plt
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(background_color='white', width=1920, height=1080).generate_from_frequencies(data)
    # wordcloud = WordCloud(max_font_size=40, background_color='white').generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    if write_to_file:
        directory = "plots/topics_{}".format(total_topics)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory+"/topic_nr_{}.png".format(topic_nr))
    else:
        plt.show()


if __name__ == "__main__":
    # build a wordcloud for each topic, requires that a dict and model has been created before
    t = TextModel()
    num_topics = 90
    t.load_dict_topics(num_topics)

    t.load_lda_model(num_topics)
    for topic in range(num_topics):
        string = t.lda.print_topic(topic, topn=50)
        print(string)
        words = [word.split("*") for word in string.split("+")]

        res = dict()
        for word in words:
            if float(word[0]) > 0:
                res[re.sub("\"", "", word[1])] = float(word[0])
        build_wordcloud(res, topic, num_topics, write_to_file=True)
        # Util.write_topic_to_json(res, topic, num_topics)



