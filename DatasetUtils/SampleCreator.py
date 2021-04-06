import pandas as pd

from Learning.LearningUtils import get_dataset, get_raw_dataset
from Utility.CSVUtils import load_data_from_CSV, save_df_as_csv
from Utility.Util import get_root_directory


def create_sample(data, size):
    """
    sorts the data by creation date and creates a random sample of size 'size'. 
    Since distribution is relatively constant the sample will keep the class distribution.
    :param data: 
    :param size: 
    :return: 
    """
    data = data.sort_values('tweet__created_at')
    rows = len(data)

    frac = size / rows
    return data.sample(frac=frac)


def create_sample_indices(size):
    """
    Sorts the data by creation date and picks a random sample of size 'size'.
    Saves the sample indices which can then be joined with the datasets for each learner.
    :param size: 
    :return: 
    """
    data = load_data_from_CSV(get_root_directory() + '/data/data_set_tweet_user_features.csv')
    data = data.reset_index(drop=True)
    data = data.sort_values('tweet__created_at')
    rows = len(data)

    frac = size / rows
    sample = data.sample(frac=frac)
    print(sample['tweet__fake'].value_counts())

    sample_indicies = pd.DataFrame(index=sample.index)
    save_df_as_csv(sample_indicies, get_root_directory() + '/data/sample/sample_indices_{}.csv'.format(size))


def get_index_of_tweet(key_id):
    data = get_raw_dataset()
    y = data.loc[data['tweet__key_id'] == key_id]
    print(y.index)
    print(y['tweet__additional_preprocessed_text'])


def use_index_as_holdout(index, data):
    """
    retruns a tweet with index 'index' as testing instance, 
    the rest of the data is returned as training set 
    :param index: 
    :param clf: 
    :return: train, test
    """
    test = data.iloc[[index]]
    train = data.drop(data.index[[index]])
    # print(test)
    # print(data.shape)
    return train, test


def get_sample(clf_name, size):
    """
    creates a sample of the data with 'size' for a given classifier. 
    Requires to build the sample beforehand.
    Uses an index list to use the same instances across the 
    different datasets from the different classifiers.
    :param clf_name: name of the classifier for which the dataset should be sampled
    :return: dataset sample
    """
    data = get_dataset(clf_name)
    data = data.reset_index(drop=True)

    indices = load_data_from_CSV(get_root_directory() + '/data/sample/sample_indices_{}.csv'.format(size))
    res = pd.concat([data, indices], axis=1, join='inner')
    print("Sample created with shape: {}".format(res.shape))
    return res


if __name__ == "__main__":
    create_sample_indices(100000)