from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from DatasetUtils.SampleCreator import get_sample
from FeatureSelection.SelectionUtils import get_best_feature_set
from Learning.LearningMain import perform_x_val
from Learning.LearningUtils import get_base_learners, get_kernel_approx, conf_matr_to_list
from ParameterTuning.TuningUtils import save_result
from Utility.TimeUtils import TimeUtils


def perform_grid_search_svm(all):

    clf_name = 'svm'
    data = get_sample(clf_name, 20000)

    features = get_best_feature_set(clf_name, all=all)

    cs = [2 ** -5, 2 ** -3, 2 ** -1, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7, 2 ** 9, 2 ** 11, 2 ** 13, 2 ** 15]
    gammas = [2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 1, 2 ** 3]
    # gammas = [2 ** -15, 2 ** -12, 2 ** -9, 2 ** -6, 2 ** -3, 2 ** -0, 2 ** 3]
    # cs = [2 ** -5, 2 ** -2, 2 ** 1, 2 ** 4, 2 ** 7, 2 ** 10, 2 ** 13, 2 ** 16]
    kernels = ['nystroem']

    # linear kernel
    for c in cs:
        clf = get_base_learners(clf_name)
        clf.set_params(**{"C":c, "dual":False})
        print("C: {}".format(c))
        f1, conf_matr = perform_x_val(data, clf, features, standardize=True)

        params = clf.get_params()
        config = dict()
        config['params'] = params
        config['f1'] = f1
        config['conf_matr'] = conf_matr_to_list(conf_matr)
        config['all'] = all
        config['kernel'] = 'linear'
        save_result(config, 'svm_20k')

    # linear kernel - class_weight balanced
    for c in cs:
        clf = get_base_learners(clf_name)
        clf.set_params(**{'C':c, 'dual':False, 'class_weight':'balanced'})
        print("C: {} balanced".format(c))
        f1, conf_matr = perform_x_val(data, clf, features, standardize=True)
        params = clf.get_params()
        config = dict()
        config['params'] = params
        config['f1'] = f1
        config['conf_matr'] = conf_matr_to_list(conf_matr)
        config['all'] = all
        config['kernel'] = 'linear'
        save_result(config, 'svm_20k')

    # approx. kernel
    for c in cs:
        for gamma in gammas:
            for kernel in kernels:
                clf = get_kernel_approx(gamma=gamma, features=features, approx=kernel)
                clf.set_params(**{"svm__C": c, "svm__dual": False})
                print("C: {}, gamma: {}".format(c, gamma))
                f1, conf_matr = perform_x_val(data, clf, features, standardize=True)
                params = clf.get_params()
                params.pop('svm', None)
                params.pop('feature_map', None)
                params.pop('steps', None)
                config = dict()
                config['params'] = params
                config['f1'] = f1
                config['conf_matr'] = conf_matr_to_list(conf_matr)
                config['all'] = all
                config['kernel'] = kernel
                save_result(config, 'svm_20k')

    # approx. kernel - class_weight balanced
    for c in cs:
        for gamma in gammas:
            for kernel in kernels:
                clf = get_kernel_approx(gamma=gamma, features=features, approx=kernel)
                clf.set_params(**{"svm__C": c, "svm__dual": False, "svm__class_weight":'balanced'})
                print("C: {}, gamma: {} balanced".format(c, gamma))
                f1, conf_matr = perform_x_val(data, clf, features, standardize=True)
                params = clf.get_params()
                params.pop('svm', None)
                params.pop('feature_map', None)
                params.pop('steps', None)
                config = dict()
                config['params'] = params
                config['f1'] = f1
                config['conf_matr'] = conf_matr_to_list(conf_matr)
                config['all'] = all
                config['kernel'] = kernel
                save_result(config, 'svm_20k')


def perform_grid_search_random_forest(all, n_estimators=1, early_stopping_rounds=10, steps=10):
    """
    peforms a grid search to find the best number of trees. Better set the number of threads to 1, 
    since RandomForestClassifier is not thread save
    :param all: 0 -> all features, 1 -> tweet features only
    :param n_estimators initial number of estimators
    :param early_stopping_rounds stop after F1 did not increase in the last n rounds
    :param steps: step size to increase n_estimators
    :return: 
    """
    clf_name = 'rf'
    data = get_sample(clf_name, 20000)
    features = get_best_feature_set(clf_name, all=all)

    f1s = list()
    for rounds in range(n_estimators, 1000, steps):
        clf = RandomForestClassifier(n_estimators=rounds)
        print('---Evaluate-{}-rounds-------------------------------------'.format(rounds))
        params = clf.get_params()
        timebefore = TimeUtils.get_time()

        f1, conf_matr = perform_x_val(data=data, clf=clf, features=features, standardize=False)

        timeafter = TimeUtils.get_time()
        time_diff = timeafter - timebefore
        conf = dict()
        conf['params'] = params
        conf['f1'] = f1
        conf['conf_matr'] = conf_matr_to_list(conf_matr)
        conf['time_diff'] = str(time_diff)
        conf['all'] = all

        save_result(filename='rf', config=conf)
        f1s.append(f1)

        if len(f1s) >= early_stopping_rounds:
            last_n = f1s[-(early_stopping_rounds - 1):]

            cont = False
            for n in last_n:
                if n > f1s[-10]:
                    cont = True
            if not cont:
                print("F1 did not increase in the last {} rounds".format(early_stopping_rounds))
                break

    # #----class-weight-balanced--------------------------------------------------------------------------
    # f1s = list()
    # for rounds in range(n_estimators, 1001, steps):
    #     clf = RandomForestClassifier(n_estimators=rounds, class_weight='balanced')
    #     print('---Evaluate-{}-rounds-------------------------------------'.format(rounds))
    #     params = clf.get_params()
    #     timebefore = TimeUtils.get_time()
    #
    #     f1, conf_matr = perform_x_val(data=data, clf=clf, features=features, standardize=False)
    #
    #     timeafter = TimeUtils.get_time()
    #     time_diff = timeafter - timebefore
    #     conf = dict()
    #     conf['params'] = params
    #     conf['f1'] = f1
    #     conf['conf_matr'] = conf_matr_to_list(conf_matr)
    #     conf['time_diff'] = str(time_diff)
    #     conf['all'] = all
    #
    #     save_result(filename='rf', config=conf)
    #     f1s.append(f1)
    #
    #     if len(f1s) >= early_stopping_rounds:
    #         last_n = f1s[-(early_stopping_rounds - 1):]
    #
    #         cont = False
    #         for n in last_n:
    #             if n > f1s[-10]:
    #                 cont = True
    #         if not cont:
    #             print("F1 did not increase in the last {} rounds".format(early_stopping_rounds))
    #             break


def perform_grid_search_xgb(all, n_estimators=1, early_stopping_rounds=10, steps=10):
    """
    peforms a grid search to find the best number of trees 
    :param all: 0 -> all features, 1 -> tweet features only
    :param n_estimators initial number of estimators
    :param early_stopping_rounds stop after F1 did not increase in the last n rounds
    :param steps: step size to increase n_estimators
    :return: 
    """
    clf_name = 'xgb'
    data = get_sample(clf_name, 20000)
    features = get_best_feature_set(clf_name, all=all)

    f1s = list()
    for rounds in range(n_estimators, 1001, steps):
        clf = XGBClassifier(n_estimators=rounds)
        print('---Evaluate-{}-rounds-------------------------------------'.format(rounds))
        params = clf.get_params()
        timebefore = TimeUtils.get_time()

        f1, conf_matr = perform_x_val(data=data, clf=clf, features=features, standardize=False)

        timeafter = TimeUtils.get_time()
        time_diff = timeafter - timebefore
        conf = dict()
        conf['params'] = conf
        conf['f1'] = f1
        conf['conf_matr'] = conf_matr_to_list(conf_matr)
        conf['time_diff'] = str(time_diff)
        conf['all'] = all

        save_result(filename='xgb_n_estimators', config=conf)
        f1s.append(f1)

        if len(f1s) >= early_stopping_rounds:
            last_n = f1s[-(early_stopping_rounds - 1):]

            cont = False
            for n in last_n:
                if n > f1s[-10]:
                    cont = True
            if not cont:
                print("F1 did not increase in the last {} rounds".format(early_stopping_rounds))
                break

if __name__ == "__main__":
    # read_result_file('svm_var_thresh_standardized', all=0)
    # data = get_dataset('svm', sample=True)[:1000]
    # features = get_feature_selection(data, all=1)
    # f1, conf_matr = perform_x_val(data, LinearSVC(class_weight='balanced', dual=False), features)
    # save_result(all=1, clf=get_base_learners('svm'), f1=f1, conf_matr=conf_matr, config={'standard':True}, filename='svm')
    # grid_search_random_forest(all=0)
    # grid_search_random_forest(all=1)

    perform_grid_search_svm(all=1)
    # perform_grid_search_svm(all=0)
    # perform_grid_search_random_forest(all=1)
    # perform_grid_search_random_forest(all=0)
    # perform_grid_search_xgb(all=1)
    # perform_grid_search_xgb(all=0)