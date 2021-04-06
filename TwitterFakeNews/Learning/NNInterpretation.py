import operator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.externals import joblib
from Learning.LearningUtils import get_learner_and_features


def load_MLP(all):
    return joblib.load('models/MLPClassifier_for_test_{}.pkl'.format(all))


def get_first_layer_weights(all, print_weights=False):
    learner, features = get_learner_and_features(clf_name='nn', all=all)
    clf = load_MLP(all)
    layer_0_weights = clf.coefs_[0]

    weights_sum = dict()
    for i in range(len(layer_0_weights)):
        weights_sum[features[i]] = np.absolute(layer_0_weights[i]).sum()

    res = sorted(weights_sum.items(), key=operator.itemgetter(1), reverse=True)

    if print_weights:
        for i, r in enumerate(res,1):
            print("{}. {}: {}".format(i, r[0], r[1]))
    return res


def plot_first_layer_weights(map, top_n=20):
    """
    plots the relative feature importance
    :param map: map with feature importances
    :param top_n: top n features
    :return: 
    """

    map = sorted(map, key=lambda tup: tup[1])
    map = map[len(map)-top_n:]
    features = [x[0] for x in map]
    scores = [x[1] for x in map]

    fig, ax1 = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.115, right=15)
    fig.canvas.set_window_title('Eldorado K-8 Fitness Chart')
    pos = np.arange(len(features))

    rects = ax1.barh(pos, [scores[k] for k in range(len(scores))],
                     align='center',
                     height=0.2, color='b',
                     tick_label=features)

    ax1.set_title("Sum of absolute first layer weights")

    ax1.set_xlim([0, scores[len(scores)-1]+0.01])
    ax1.xaxis.set_major_locator(MaxNLocator(11))
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)

    # set X-axis tick marks at the deciles
    imp = ax1.text(.5, -.07, 'Sum of absolute weights',
                            horizontalalignment='center', size='small',
                            transform=ax1.transAxes)


    for rect in rects:
        ax1.text(rect.get_width() + 0.0002, rect.get_y() + rect.get_height()/2.,
                '{}'.format(rect.get_width()),
                ha='left', va='center')

    plt.show()

if __name__ == '__main__':
    all = 1
    abs_weights = get_first_layer_weights(all=all, print_weights=True)

    for i, res in enumerate(abs_weights, 1):
        if 'sentiment' in res[0]:
            print("{}. {} ".format(i, res))


    # plot_first_layer_weights(abs_weights, top_n=20)
    # clf = load_MLP(all)
    pass
