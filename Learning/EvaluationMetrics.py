from Learning.LearningUtils import conf_matr_to_list


def calc_precision(conf_matr):
    if (conf_matr[0, 0] + conf_matr[1, 0]) == 0:
        return 0.
    else:
        return conf_matr[0, 0] / (conf_matr[0, 0] + conf_matr[1, 0])


def calc_recall(conf_matr):
    if (conf_matr[0, 0] + conf_matr[0, 1]) == 0:
        return 0.
    else:
        return conf_matr[0, 0] / (conf_matr[0, 0] + conf_matr[0, 1])


def calc_accuracy(conf_matr):
    return (conf_matr[0, 0] + conf_matr[1, 1]) / (
        conf_matr[0, 0] + conf_matr[1, 0] + conf_matr[0, 1] + conf_matr[1, 1])


def calc_f1(conf_matr):
    if (calc_precision(conf_matr) + calc_recall(conf_matr)) == 0:
        return 0
    else:
        return (2 * calc_precision(conf_matr) * calc_recall(conf_matr)) / (calc_precision(conf_matr) + calc_recall(conf_matr))

def print_result(conf_matr):
    print(conf_matr)
    print("Accuracy: " + str(calc_accuracy(conf_matr)))
    print("Precision: " + str(calc_precision(conf_matr)))
    print("Recall: " + str(calc_recall(conf_matr)))
    print("F1: " + str(calc_f1(conf_matr)))

def get_result(conf_matr):
    conf = dict()
    conf['conf_matr'] = conf_matr_to_list(conf_matr)
    conf['all'] = all

    res = dict()
    res['conf_matr'] = conf_matr_to_list(conf_matr)
    res['acc'] = calc_accuracy(conf_matr)
    res['prec'] = calc_precision(conf_matr)
    res['rec'] = calc_recall(conf_matr)
    res['f1'] = calc_f1(conf_matr)
    return res