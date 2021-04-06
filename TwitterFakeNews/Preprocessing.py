import pandas as pd
from collections import Counter
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from FeatureEngineering.FeatureSelector import get_feature_selection


class Preprocessor:

    norm_pickle_folder = '../data/normalization/'

    @staticmethod
    def replace_missing_possibly_sensitive(X):
        """
        replaces all missing values in 'tweet__truncated'
        :param X: 
        :return: 
        """
        X['tweet__possibly_sensitive'].fillna(2, inplace=True)
        return X

    @staticmethod
    def impute_possibly_sensitive(data):
        print(data['tweet__possibly_sensitive'].value_counts())
        sel = get_feature_selection(data)
        sel.append("tweet__fake")

        data = data[sel]
        print([data.columns])

        sel.remove('tweet__possibly_sensitive')
        X_tmp = data.dropna()
        X_train = X_tmp[sel]
        y_train = X_tmp['tweet__possibly_sensitive']


        X_test = data.loc[data['tweet__possibly_sensitive'].isnull()][sel]

        nbrs = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
        y_test = nbrs.predict(X_test)

        print(Counter(y_test))

    @staticmethod
    def apply_col_min_max(data, filename):
        """
        loads a normalization model from a file and applies it to the data
        :param data: 
        :param filename: file to normalization model
        :return: 
        """
        cols = data.columns
        x = data.values
        min_max_scaler = joblib.load(Preprocessor.norm_pickle_folder+filename)
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        df.columns = cols
        return df

    @staticmethod
    def normalize_min_max(data, axis=0, name=None):
        """
        performs min_max scaling
        :param data: data to normalize
        :param axis: 0: columns, 1: rows
        :param name: if a name is given, min_max_model is stored
        :return: 
        """
        if axis == 0:
            # column normalization
            cols = data.columns
            x = data.values
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            if name is None:
                x_scaled = min_max_scaler.fit_transform(x)
            else:
                min_max_scaler.fit(x)
                joblib.dump(min_max_scaler, Preprocessor.norm_pickle_folder+'min_max_'+name+'.pkl')
                x_scaled = min_max_scaler.transform(x)
            df = pd.DataFrame(x_scaled)
            df.columns = cols
            return df

        else:
            # row normalization
            cols = data.columns
            minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(data.T)
            X_minmax = minmax_scale.transform(data.T).T
            df = pd.DataFrame(X_minmax)
            df.columns = cols
            return df
