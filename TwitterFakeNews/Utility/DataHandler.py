from pandas import HDFStore
from FeatureEngineering.FeatureSelector import get_mixed_features

class DataHandler:

    NAME = 'twitter_data_original_new'

    @staticmethod
    def get_store(read_write):
        return HDFStore('../FeatureEngineering/twitter_data.h5', mode=read_write)


    @staticmethod
    def load_data(name=NAME):
        """loads the data set"""
        store = DataHandler.get_store('r')
        # print(store)
        print(store.keys())
        print("Load data set...")
        # return store[name]
        store_to_return = store[name] # load it
        store.close()
        return store_to_return
        # return read_hdf('twitter_data.h5', 'twitter_data',
        #          where=['A>.5'], columns=['A', 'B'])
        # return read_hdf('twitter_data.h5', name)

    @staticmethod
    def store_data(data, mode='w', name=NAME):
        """stores data. Default open mode: 'w'"""
        store = DataHandler.get_store(mode)
        try:
            print("...store data set.")
            # print(data.head())
            columns = get_mixed_features()
            columns_in_data = list()
            # for col in data.columns:
            #     print("{} {}".format(data[col].dtype, col))
            #     if data[col].dtype == np.object:
            #         print("{} contains mixed integer and was converted to string".format(col))
            #         data[col] = data[col].astype('str')

            for c in columns:
                if c in data.columns:
                    columns_in_data.append(c)
            if columns_in_data:
                data.loc[:, columns_in_data] = data[columns_in_data].applymap(str)
            print(data.shape)
            store.put(name, data, format='table', data_columns=True)  # save it
            store.close()
        except Exception as e:
            print("Error - store_data: {}".format(e))
            store.close()

