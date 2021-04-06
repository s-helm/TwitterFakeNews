from Database.DatabaseHandler import DatabaseHandler
from Utility.CSVUtils import write_data_to_CSV
from Utility.TimeUtils import TimeUtils


def retrieve_dataset(testset):
    """
    retrieves the dataset from the database
    :param testset: if true the testset is retrieved from the database
    :return: 
    """
    time = TimeUtils.get_time()
    if testset:
        dataset = DatabaseHandler.load_data_set(testset=True)
        name = "testset_" + str(time.day) + "_" + str(time.month) + ".csv"
    else:
        dataset = DatabaseHandler.load_data_set(testset=True)
        name = "data_set_" + str(time.day) + "_" + str(time.month) + ".csv"
    write_data_to_CSV(dataset, name)

if __name__ == '__main__':
    retrieve_dataset(testset=True)
