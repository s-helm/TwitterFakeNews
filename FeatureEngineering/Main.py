from FeatureEngineering.FeatureCreator import create_features
from Utility.CSVUtils import load_data_from_CSV


# data = load_data_from_CSV('data_set_7_7_sample.csv')
# data = data.reset_index(drop=True)
#
# print(data.index)
# trigram_vectors = find_frequent_pos_trigrams(data, min_doc_frequency=1000, no_above=0.4, keep_n=100)
# for key, vector in trigram_vectors.items():
#     data['tweet__contains_pos_trigram_{}'.format(re.sub(" ", "_", str(key)))] = vector
#
# cols = [col for col in data.columns if 'contains_pos_trigram' in col or col == 'tweet__id']
# print(cols)
# print(data.shape)
# save_df_as_csv(data[cols], 'pos_trigrams.csv')

data = load_data_from_CSV("..some/data.csv")
data = create_features(data)





