#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from scipy.stats import norm
import copy

import scipy.optimize as opt
# importing bayes doesn't work in BRUTE :(, please copy the functions into this file


def minimax_strategy_discrete(distribution1, distribution2):
    """
    q = minimax_strategy_discrete(distribution1, distribution2)

    Find the optimal Minimax strategy for 2 discrete distributions.

    :param distribution1:           pXk(x|class1) given as a (n, n) np array
    :param distribution2:           pXk(x|class2) given as a (n, n) np array
    :return q:                      optimal strategy, (n, n) np array, values 0 (class 1) or 1 (class 2)
    :return: opt_i:                 index of the optimal solution found during the search, Python int in [0, n*n + 1) range
    :return: eps1:                  cumulative error on the first class for all thresholds, (n * n + 1,) numpy array
    :return: eps2:                  cumulative error on the second class for all thresholds, (n * n + 1,) numpy array
    """
    n = distribution1.shape[0]
    d1 = distribution1.flatten()
    d2 = distribution2.flatten()
    
    # likelihood ratio (LR)
    r = np.divide(d1, d2)

    # sort distributions by LR
    i = np.argsort(r)
    d1 = d1[i]
    d2 = d2[i]

    # calculate classification errors
    eps1 = np.cumsum(d1)
    eps1 = np.append([0], eps1)
    eps2 = np.cumsum(d2[::-1])[::-1]
    eps2 = np.append(eps2, [0])
   
    # decide strategy
    opt_i = np.argmin(np.maximum(eps1, eps2))
    q = np.zeros(n*n, dtype=int)
    q[i[:opt_i]] = 1
    q = q.reshape(n,n)
    
    return q, opt_i, eps1, eps2


def classify_discrete(imgs, q):
    """
    function label = classify_discrete(imgs, q)

    Classify images using discrete measurement and strategy q.

    :param imgs:    test set images, (h, w, n) uint8 np array
    :param q:       strategy (21, 21) np array of 0 or 1
    :return:        image labels, (n, ) np array of 0 or 1
    """
    x = compute_measurement_lr_discrete(imgs)
    y = compute_measurement_ul_discrete(imgs)

    X = Y = np.arange(-10,11)
    idx = np.searchsorted(X, x)
    idy = np.searchsorted(Y, y)

    labels = q[idx, idy]

    return labels


def worst_risk_cont(distribution_A, distribution_B, true_A_prior):
    """
    Find the optimal bayesian strategy for true_A_prior (assuming 0-1 loss) and compute its worst possible risk in case the priors are different.

    :param distribution_A:          parameters of the normal dist.
                                    distribution_A['Mean'], distribution_A['Sigma'] - python floats
    :param distribution_B:          the same as distribution_A
    :param true_A_prior:            true A prior probability - python float
    :return worst_risk:             worst possible bayesian risk when evaluated with different prior
    """
    distribution_A['Prior'] = true_A_prior
    distribution_B['Prior'] = 1 - true_A_prior
    priors = np.array([0,1], dtype=float)

    q = find_strategy_2normal(distribution_A, distribution_B)
    risks = risk_fix_q_cont(distribution_A, distribution_B, priors, q)
    worst_risk = max(risks)
    return worst_risk


def minimax_strategy_cont(distribution_A, distribution_B):
    """
    q, worst_risk = minimax_strategy_cont(distribution_A, distribution_B)

    Find minimax strategy.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'] - python floats
    :param distribution_B:  the same as distribution_A
    :return q:              strategy dict - see bayes.find_strategy_2normal
                               q['t1'], q['t2'] - decision thresholds - python floats
                               q['decision'] - (3, ) np.int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return worst_risk      worst risk of the minimax strategy q - python float
    """
    helper_func = lambda prior: worst_risk_cont(distribution_A, distribution_B, prior)
    opt_p = opt.fminbound(helper_func, 0.0, 1.0)

    distribution_A['Prior'] = float(opt_p)
    distribution_B['Prior'] = float(1.0 - opt_p)

    q = find_strategy_2normal(distribution_A, distribution_B)
    worst_risk = bayes_risk_2normal(distribution_A, distribution_B, q)
    return q, worst_risk


def risk_fix_q_cont(distribution_A, distribution_B, distribution_A_priors, q):
    """
    Computes bayesian risks for fixed strategy and various priors.

    :param distribution_A:          parameters of the normal dist.
                                    distribution_A['Mean'], distribution_A['Sigma'] - python floats
    :param distribution_B:          the same as distribution_A
    :param distribution_A_priors:   priors (n, ) np.array
    :param q:                       strategy dict - see bayes.find_strategy_2normal
                                       q['t1'], q['t2'] - decision thresholds - python floats
                                       q['decision'] - (3, ) np.int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return risks:                  bayesian risk of the strategy q with varying priors (n, ) np.array
    """
    risks = np.zeros_like(distribution_A_priors)
    for i in range(distribution_A_priors.size):
        distribution_A['Prior'] = float(distribution_A_priors[i])
        distribution_B['Prior'] = float(1 - distribution_A['Prior'])
        risks[i] = bayes_risk_2normal(distribution_A, distribution_B, q)
    return risks


################################################################################
#####                                                                      #####
#####                Put functions from previous labs here.                #####
#####            (Sorry, we know imports would be much better)             #####
#####                                                                      #####
################################################################################

def classification_error(predictions, labels):
    """
    error = classification_error(predictions, labels)

    :param predictions: (n, ) np.array of values 0 or 1 - predicted labels
    :param labels:      (n, ) np.array of values 0 or 1 - ground truth labels
    :return:            error - classification error ~ a fraction of predictions being incorrect
                        python float in range <0, 1>
    """
    comparison = (predictions != labels).astype(int)
    error = (np.sum(comparison) / labels.shape[0]).astype(float)
    return error


def find_strategy_2normal(distribution_A, distribution_B):
    """
    q = find_strategy_2normal(distribution_A, distribution_B)

    Find optimal bayesian strategy for 2 normal distributions and zero-one loss function.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'], distribution_A['Prior'] - python floats
    :param distribution_B:  the same as distribution_A

    :return q:              strategy dict
                               q['t1'], q['t2'] - decision thresholds - python floats
                               q['decision'] - (3, ) np.int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
                               If there is only one threshold, q['t1'] should be equal to q['t2'] and the middle decision should be 0
                               If there is no threshold, q['t1'] and q['t2'] should be -/+ infinity and all the decision values should be the same
                                (0 preferred if both strategies would have the same risk)
    """
    s_A = distribution_A['Sigma']
    m_A = distribution_A['Mean']
    p_A = distribution_A['Prior']
    s_B = distribution_B['Sigma']
    m_B = distribution_B['Mean']
    p_B = distribution_B['Prior']

    q = {'t1': float(-np.inf),
         't2': float(np.inf),
         'decision': np.zeros((3, ), dtype=np.int32)
         }

    # extreme priors
    eps = 1e-10
    if p_A < eps:
        q['decision'] = np.array((1,1,1), dtype=np.int32)
    elif p_B < eps:
        q['decision'] = np.array((0,0,0), dtype=np.int32)
    elif abs(p_A - p_B) < eps:
        q['decision'] = np.array((0,0,0), dtype=np.int32)
    else:
        a = s_A **2 - s_B **2
        b = 2 * (m_A * s_B **2 - m_B * s_A **2)
        c = s_A **2 * m_B **2 - s_B **2 * m_A **2 - 2 * s_A **2 * s_B **2 * np.log((p_B * s_A) / (p_A * s_B))
        if a == 0:
            # same sigmas -> not quadratic
            if b == 0:
                # same sigmas and same means -> not even linear
                if p_A > p_B:
                    q['decision'] = np.array((0,0,0), dtype=np.int32)
                else:
                    q['decision'] = np.array((1,1,1), dtype=np.int32)
            else:
                # same sigmas, different means -> linear equation
                q['t1'] = q['t2'] = float(-c/b)
                if b < 0:
                    q['decision'] = np.array((0,0,1), dtype=np.int32)
                else:
                    q['decision'] = np.array((1,0,0), dtype=np.int32)
        else:
            # quadratic equation
            D = float(b**2 - 4*a*c)
            if D > 0:
                t1 = float((-b - np.sqrt(D)) / (2*a))
                t2 = float((-b + np.sqrt(D)) / (2*a))
                q['t1'] = min(t1, t2)
                q['t2'] = max(t1, t2)
                if a < 0:
                    q['decision'] = np.array((1,0,1), dtype=np.int32)
                else:
                    q['decision'] = np.array((0,1,0), dtype=np.int32)
            elif D == 0:
                q['t1'] = q['t2'] = float(-b / 2*a)
                q['decision'] = np.array((1,0,1), dtype=np.int32)
            elif D < 0:
                if a > 0:
                    q['decision'] = np.array((0,0,0), dtype=np.int32)
                else:
                    q['decision'] = np.array((1,1,1), dtype=np.int32)

    return q


def bayes_risk_2normal(distribution_A, distribution_B, q):
    """
    R = bayes_risk_2normal(distribution_A, distribution_B, q)

    Compute bayesian risk of a strategy q for 2 normal distributions and zero-one loss function.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'], distribution_A['Prior'] python floats
    :param distribution_B:  the same as distribution_A
    :param q:               strategy
                               q['t1'], q['t2'] - float decision thresholds (python floats)
                               q['decision'] - (3, ) np.int32 np.array 0/1 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return:    R - bayesian risk, python float
    """
    X1 = distribution_A if q['decision'][0] == 0 else distribution_B
    X2 = distribution_A if q['decision'][1] == 0 else distribution_B
    X3 = distribution_A if q['decision'][2] == 0 else distribution_B

    R = 1 - (norm.cdf(q['t1'], loc=X1['Mean'], scale=X1['Sigma']) * X1['Prior'] + \
            (norm.cdf(q['t2'], loc=X2['Mean'], scale=X2['Sigma']) - norm.cdf(q['t1'], loc=X2['Mean'], scale=X2['Sigma'])) * X2['Prior'] + \
            (1 - norm.cdf(q['t2'], loc=X3['Mean'], scale=X3['Sigma'])) * X3['Prior'])
    R = float(R)
    return R


def classify_2normal(measurements, q):
    """
    label = classify_2normal(measurements, q)

    Classify images using continuous measurements and strategy q.

    :param imgs:    test set measurements, np.array (n, )
    :param q:       strategy
                    q['t1'] q['t2'] - float decision thresholds
                    q['decision'] - (3, ) int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return:        label - classification labels, (n, ) int32
    """
    label = np.piecewise(measurements, [measurements <= q['t1'], (measurements > q['t1']) & \
            (measurements <= q['t2']), measurements > q['t2']], q['decision'])
    return label


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


def plot_lr_threshold(eps1, eps2, thr):
    """
    Plot the search for the strategy

    :param eps1:  cumulative error on the first class for all thresholds, (N + 1, ) numpy array
    :param eps2:  cumulative error on the second class for all thresholds, (N + 1, ) numpy array
    :param thr:   index of the optimal solution found during the search, Python int in [0, N+1) range
    :return:      matplotlib.pyplot figure
    """

    fig = plt.figure(figsize=(15, 5))
    plt.plot(eps2, 'o-', label='$\epsilon_2$')
    plt.plot(eps1, 'o-', label='$\epsilon_1$')
    plt.plot([thr, thr], [-0.02, 1], 'k')
    plt.legend()
    plt.ylabel('classification error')
    plt.xlabel('i')
    plt.title('minimax - LR threshold search')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # inset axes....
    ax = plt.gca()
    axins = ax.inset_axes([0.4, 0.2, 0.4, 0.6])
    axins.plot(eps2, 'o-')
    axins.plot(eps1, 'o-')
    axins.plot([thr, thr], [-0.02, 1], 'k')
    axins.set_xlim(thr - 10, thr + 10)
    axins.set_ylim(-0.02, 1)
    axins.xaxis.set_major_locator(MaxNLocator(integer=True))
    axins.set_title('zoom in')
    # ax.indicate_inset_zoom(axins)

    return fig


def plot_discrete_strategy(q, letters):
    """
    Plot for discrete strategy

    :param q:        strategy (21, 21) np array of 0 or 1
    :param letters:  python string with letters, e.g. 'CN'
    :return:         matplotlib.pyplot figure
    """
    fig = plt.figure()
    im = plt.imshow(q, extent=[-10,10,10,-10])
    values = np.unique(q)   # values in q
    # get the colors of the values, according to the colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [ mpatches.Patch(color=colors[i], label="Class {}".format(letters[values[i]])) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('X')
    plt.xlabel('Y')

    return fig


def compute_measurement_lr_cont(imgs):
    """
    x = compute_measurement_lr_cont(imgs)

    Compute measurement on images, subtract sum of right half from sum of
    left half.

    :param imgs:    set of images, (h, w, n) numpy array
    :return x:      measurements, (n, ) numpy array
    """
    assert len(imgs.shape) == 3

    width = imgs.shape[1]
    sum_rows = np.sum(imgs, dtype=np.float64, axis=0)

    x = np.sum(sum_rows[0:int(width / 2),:], axis=0) - np.sum(sum_rows[int(width / 2):,:], axis=0)

    assert x.shape == (imgs.shape[2], )
    return x


def compute_measurement_lr_discrete(imgs):
    """
    x = compute_measurement_lr_discrete(imgs)

    Calculates difference between left and right half of image(s).

    :param imgs:    set of images, (h, w, n) (or for color images (h, w, 3, n)) np array
    :return x:      measurements, (n, ) np array of values in range <-10, 10>,
    """
    assert len(imgs.shape) in (3, 4)
    assert (imgs.shape[2] == 3 or len(imgs.shape) == 3)

    mu = -563.9
    sigma = 2001.6

    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=2)

    imgs = imgs.astype(np.int32)
    height, width, channels, count = imgs.shape

    x_raw = np.sum(np.sum(np.sum(imgs[:, 0:int(width / 2), :, :], axis=0), axis=0), axis=0) - \
            np.sum(np.sum(np.sum(imgs[:, int(width / 2):, :, :], axis=0), axis=0), axis=0)
    x_raw = np.squeeze(x_raw)

    x = np.atleast_1d(np.round((x_raw - mu) / (2 * sigma) * 10))
    x[x > 10] = 10
    x[x < -10] = -10

    assert x.shape == (imgs.shape[-1], )
    return x


def compute_measurement_ul_discrete(imgs):
    """
    x = compute_measurement_ul_discrete(imgs)

    Calculates difference between upper and lower half of image(s).

    :param imgs:    set of images, (h, w, n) (or for color images (h, w, 3, n)) np array
    :return x:      measurements, (n, ) np array of values in range <-10, 10>,
    """
    assert len(imgs.shape) in (3, 4)
    assert (imgs.shape[2] == 3 or len(imgs.shape) == 3)

    mu = -563.9
    sigma = 2001.6

    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=2)

    imgs = imgs.astype(np.int32)
    height, width, channels, count = imgs.shape

    x_raw = np.sum(np.sum(np.sum(imgs[0:int(height / 2), :, :, :], axis=0), axis=0), axis=0) - \
            np.sum(np.sum(np.sum(imgs[int(height / 2):, :, :, :], axis=0), axis=0), axis=0)
    x_raw = np.squeeze(x_raw)

    x = np.atleast_1d(np.round((x_raw - mu) / (2 * sigma) * 10))
    x[x > 10] = 10
    x[x < -10] = -10

    assert x.shape == (imgs.shape[-1], )
    return x


def create_test_set(images_test, labels_test, letters, alphabet):
    """
    images, labels = create_test_set(images_test, letters, alphabet)

    Return subset of the <images_test> corresponding to <letters>

    :param images_test: test images of all letter in alphabet - np.array (h, w, n)
    :param labels_test: labels for images_test - np.array (n,)
    :param letters:     python string with letters, e.g. 'CN'
    :param alphabet:    alphabet used in images_test - ['A', 'B', ...]
    :return images:     images - np array (h, w, n)
    :return labels:     labels for images, np array (n,)
    """

    images = np.empty((images_test.shape[0], images_test.shape[1], 0), dtype=np.uint8)
    labels = np.empty((0,))
    for i in range(len(letters)):
        letter_idx = np.where(alphabet == letters[i])[0]
        images = np.append(images, images_test[:, :, labels_test == letter_idx], axis=2)
        lab = labels_test[labels_test == letter_idx]
        labels = np.append(labels, np.ones_like(lab) * i, axis=0)

    return images, labels


def show_classification(test_images, labels, letters):
    """
    show_classification(test_images, labels, letters)

    create montages of images according to estimated labels

    :param test_images:     np.array (h, w, n)
    :param labels:          labels for input images np.array (n,)
    :param letters:         string with letters, e.g. 'CN'
    """
    assert isinstance(labels, np.ndarray), "'labels' must be a numpy array!"

    def montage(images, colormap='gray'):
        """
        Show images in grid.

        :param images:      np.array (h, w, n)
        :param colormap:    numpy colormap
        """
        h, w, count = np.shape(images)
        h_sq = int(np.ceil(np.sqrt(count)))
        w_sq = h_sq
        im_matrix = np.zeros((h_sq * h, w_sq * w))

        image_id = 0
        for j in range(h_sq):
            for k in range(w_sq):
                if image_id >= count:
                    break
                slice_w = j * h
                slice_h = k * w
                im_matrix[slice_h:slice_h + w, slice_w:slice_w + h] = images[:, :, image_id]
                image_id += 1
        plt.imshow(im_matrix, cmap=colormap)
        plt.axis('off')
        return im_matrix

    for i in range(len(letters)):
        imgs = test_images[:,:,labels==i]
        subfig = plt.subplot(1,len(letters),i+1)
        montage(imgs)
        plt.title(letters[i])


################################################################################
#####                                                                      #####
#####             Below this line you may insert debugging code            #####
#####                                                                      #####
################################################################################

def main():
    # HERE IT IS POSSIBLE TO ADD YOUR TESTING OR DEBUGGING CODE
    pass

if __name__ == "__main__":
    main()
