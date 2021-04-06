import functools
import random
import numpy as np
from threading import Thread
from sklearn.neural_network import MLPClassifier
from DatasetUtils.SampleCreator import get_sample
from FeatureSelection.SelectionUtils import get_best_feature_set
from Learning.LearningMain import perform_x_val
from Learning.LearningUtils import conf_matr_to_list
from ParameterTuning.TuningUtils import save_result
from Utility.TimeUtils import TimeUtils

def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco


def grid_search_layer_neurons(all):
    data = get_sample('nn', 20000)
    input_features = get_best_feature_set('nn', all)
    input_layer_size = len(input_features)
    output_layer_size = 1

    neurons = output_layer_size
    while True:
        neurons += 10
        if neurons > input_layer_size:
            neurons = input_layer_size

        timebefore = TimeUtils.get_time()
        clf_nn = MLPClassifier(hidden_layer_sizes=(neurons,))
        print("Number of neurons: {}".format(neurons))
        f1, conf_matr = perform_x_val(data=data, clf=clf_nn, features=input_features, standardize=True)

        timeafter = TimeUtils.get_time()
        time_diff = timeafter - timebefore

        conf_matr = conf_matr_to_list(conf_matr)
        config = dict()
        config['params'] = clf_nn.get_params()
        config['f1'] = f1
        config['all'] = all
        config['conf_matr'] = conf_matr
        config['time_diff'] = str(time_diff)

        use_case = ''
        if all == 1:
            use_case = 'wo_user'
        elif all == 0:
            use_case = 'with_user'

        save_result(config, 'neural_network_neurons_'+use_case)

        if neurons == input_layer_size:
            break


def random_iteration(data, features, all):
    """
    runs a neural network with a random combination of parameters
    :param all: 
    :return: 
    """
    # data = get_dataset(clf='nn', sample=True)

    # standardize = bool(random.getrandbits(1))
    hidden_units = int(round(np.exp(random.uniform(np.log(1), np.log(len(features))))))
    activation = random.choice(['logistic', 'tanh'])
    batch_size = random.choice([20, 100, 200])
    solver = random.choice(['adam', 'sgd'])
    learning_rate_init = np.exp(random.uniform(np.log(10 ** -6), np.log(10)))
    learning_rate = random.choice(['constant', 'invscaling', 'adaptive'])
    early_stopping = bool(random.getrandbits(1))
    max_iter = random.randint(100, 500)
    alpha = np.exp(random.uniform(np.log(10 ** -7), np.log(10 ** -1)))

    config = dict()
    config['hidden_units'] = hidden_units
    config['activation'] = activation
    config['batch_size'] = batch_size
    config['solver'] = solver
    config['learning_rate_init'] = learning_rate_init
    config['learning_rate'] = learning_rate
    config['early_stopping'] = early_stopping
    config['max_iter'] = max_iter
    config['alpha'] = alpha

    try:
        clf_nn = MLPClassifier(hidden_layer_sizes=(hidden_units,), activation=activation, batch_size=batch_size, solver=solver,
              learning_rate_init=learning_rate_init, learning_rate=learning_rate, max_iter=max_iter, alpha=alpha, early_stopping=early_stopping)
        run_and_save(data, clf_nn, features, all)
    except Exception as e:
        print("{} with config: ".format(e))
        print(config)
        config['all'] = all
        config['f1'] = float('nan')
        config['conf_matr'] = [float('nan'), float('nan'), float('nan'), float('nan')]
        config['time_diff'] = float('nan')
        save_result(config, 'nn')

@timeout(600)
def run_and_save(data, clf_nn, features, all):
    timebefore = TimeUtils.get_time()

    f1, conf_matr = perform_x_val(data=data, clf=clf_nn, features=features, standardize=True)

    timeafter = TimeUtils.get_time()
    time_diff = timeafter - timebefore

    conf_matr = conf_matr_to_list(conf_matr)
    res = dict()
    res['params'] = clf_nn.get_params()
    res['all'] = all
    res['f1'] = f1
    res['conf_matr'] = conf_matr
    res['time_diff'] = str(time_diff)
    save_result(res, 'nn')


def perform_random_search(iterations, all):
    """
    performs x iterations of random_iteration
    :param iterations: nr of iterations
    :param all: feature set to use
    :return: 
    """
    data = get_sample('nn', 20000)
    features = get_best_feature_set(clf_name='nn', all=all)

    for i in range(iterations):
        print("----Start-iteration:-{}---------------------------------------------------".format(i))
        random_iteration(data=data, features=features, all=all)



if __name__ == "__main__":
    start = TimeUtils.get_time()
    perform_random_search(iterations=100, all=1)
    print("Time for 100 iterations: {}".format(TimeUtils.get_time()-start))

    start = TimeUtils.get_time()
    perform_random_search(iterations=100, all=0)
    print("Time for 100 iterations: {}".format(TimeUtils.get_time()-start))

