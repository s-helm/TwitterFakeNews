from sklearn.preprocessing import StandardScaler


def standardize_data(X_train, X_test):
    """
    standardizes the data to have zero mean and unit variance.
    :param X_train: 
    :param X_test: 
    :return: 
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # apply same transformation to test data
    X_test = scaler.transform(X_test)
    return X_train, X_test