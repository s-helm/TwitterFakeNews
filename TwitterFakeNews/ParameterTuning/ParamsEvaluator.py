from Learning.LearningMain import perform_x_val
from Learning.LearningUtils import get_dataset, conf_matr_to_list, get_learner_and_features
from ParameterTuning.TuningUtils import save_result
from Utility.TimeUtils import TimeUtils


def evaluate_best_params(clf_name, all):
    """
    evaluates the best parameters on the whole dataset
    :param clf_name: name of clf
    :param all: 0: tweet+user features, 1: tweet features
    :return: 
    """
    clf, features = get_learner_and_features(clf_name=clf_name, all=all)
    data = get_dataset(clf_name)

    timebefore = TimeUtils.get_time()
    f1, conf_matr = perform_x_val(clf=clf, data=data, features=features, standardize=True)
    timeafter = TimeUtils.get_time()

    time_diff = timeafter - timebefore
    params_new = dict()
    params_new['f1'] = f1
    params_new['all'] = all
    params_new['conf_matr'] = conf_matr_to_list(conf_matr)
    params_new['time_diff'] = str(time_diff)
    params_new['params'] = clf.get_params()
    params_new['features'] = features

    save_result(filename='{}_full_set_'.format(clf_name), config=params_new)

if __name__ == '__main__':
    evaluate_best_params('svm', all=1)
    # evaluate_best_params('svm', all=0)