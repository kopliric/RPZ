#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def bayes_risk_discrete(discrete_A, discrete_B, W, q):
    """
    R = bayes_risk_discrete(discrete_A, discrete_B, W, q)

    Compute bayesian risk for a discrete strategy q

    :param discrete_A['Prob']:      pXk(x|A) given as a (n, ) np.array
    :param discrete_A['Prior']:     prior probability pK(A)
    :param discrete_B['Prob']:      pXk(x|B) given as a (n, ) np.array
    :param discrete_B['Prior']:     prior probability pK(B)
    :param W:                       cost function np.array (states, decisions)
                                    (nr. of states and decisions is fixed to 2)
    :param q:                       strategy - (n, ) np.array, values 0 or 1
    :return:                        bayesian risk - python float
    """
    joint_A = discrete_A['Prob'] * discrete_A['Prior']
    joint_B = discrete_B['Prob'] * discrete_B['Prior']
    
    R = np.sum(joint_A * W[0,q] + joint_B * W[1,q])
    R = float(R)
    return R


def find_strategy_discrete(discrete_A, discrete_B, W):
    """
    q = find_strategy_discrete(distribution1, distribution2, W)

    Find bayesian strategy for 2 discrete distributions.

    :param discrete_A['Prob']:      pXk(x|A) given as a (n, ) np.array
    :param discrete_A['Prior']:     prior probability pK(A)
    :param discrete_B['Prob']:      pXk(x|B) given as a (n, ) np.array
    :param discrete_B['Prior']:     prior probability pK(B)
    :param W:                       cost function np.array (states, decisions)
                                    (nr. of states and decisions is fixed to 2)
    :return:                        q - optimal strategy (n, ) np.array, values 0 or 1
    """
    post_A = (discrete_A['Prob'] * discrete_A['Prior']) / np.sum(discrete_A['Prob'])
    post_B = (discrete_B['Prob'] * discrete_B['Prior']) / np.sum(discrete_B['Prob'])

    R = np.array([post_A * W[0,0] + post_B * W[1,0], 
                  post_A * W[0,1] + post_B * W[1,1]])
    q = R[0] > R[1] # return 0 if A, 1 if C
    q = q.astype(int)
    return q


def classify_discrete(measurements, q):
    """
    function label = classify_discrete(measurements, q)

    Classify discrete measurement using a strategy q.

    :param measurements:    test set discrete measurements, (n, ) np.array, values from <-10, 10>
    :param q:               strategy (21, ) np.array of 0 or 1
    :return:                image labels, (n, ) np.array of 0 or 1
    """
    X = np.arange(-10,11)
    idx = np.searchsorted(X, measurements)
    label = q[idx]
    return label


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


def compute_measurement_lr_cont(imgs):
    """
    x = compute_measurement_lr_cont(imgs)

    Compute measurement on images, subtract sum of right half from sum of
    left half.

    :param imgs:    set of images, (h, w, n)
    :return:        measurements, (n, )
    """
    assert len(imgs.shape) == 3

    width = imgs.shape[1]
    sum_rows = np.sum(imgs, dtype=np.float64, axis=0)

    x = np.sum(sum_rows[0:int(width / 2), :], axis=0) - np.sum(sum_rows[int(width / 2):, :], axis=0)

    assert x.shape == (imgs.shape[2], )
    return x


def visualize_discrete(discrete_A, discrete_B, q):
    """
    visualize_discrete(discrete_A, discrete_B, q)

    Visualize a strategy for 2 discrete distributions.

    :param discrete_A['Prob']:      pXk(x|A) given as a (n, ) np.array
    :param discrete_A['Prior']:     prior probability pK(A)
    :param discrete_B['Prob']:      pXk(x|B) given as a (n, ) np.array
    :param discrete_B['Prior']:     prior probability pK(B)
    :param q:                       strategy - (n, ) np.array, values 0 or 1
    """

    posterior_A = discrete_A['Prob'] * discrete_A['Prior']
    posterior_B = discrete_B['Prob'] * discrete_B['Prior']

    max_prob = np.max([posterior_A, posterior_B])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Posterior probabilities and strategy q")
    plt.xlabel("feature")
    plt.ylabel("posterior probabilities")

    bins = np.array(range(posterior_A.size + 1)) - int(posterior_A.size / 2)

    width = 0.75
    bar_plot_A = plt.bar(bins[:-1], posterior_A, width=width, color='b', alpha=0.75)
    bar_plot_B = plt.bar(bins[:-1], posterior_B, width=width, color='r', alpha=0.75)

    plt.legend((bar_plot_A, bar_plot_B), (r'$p_{XK}(x,A)$', r'$p_{XK}(x,B)$'))

    sub_level = - max_prob / 8
    height = np.abs(sub_level)
    for idx in range(len(bins[:-1])):
        b = bins[idx]
        col = 'r' if q[idx] == 1 else 'b'
        patch = patches.Rectangle([b - 0.5, sub_level], 1, height, angle=0.0, color=col, alpha=0.75)
        ax.add_patch(patch)

    plt.ylim(bottom=sub_level)
    plt.text(bins[0], -max_prob / 16, 'strategy q')


def visualize_2norm(cont_A, cont_B, q):
    n_sigmas = 5
    n_points = 200

    A_range = (cont_A['Mean'] - n_sigmas * cont_A['Sigma'],
               cont_A['Mean'] + n_sigmas * cont_A['Sigma'])
    B_range = (cont_B['Mean'] - n_sigmas * cont_B['Sigma'],
               cont_B['Mean'] + n_sigmas * cont_B['Sigma'])
    start = min(A_range[0], B_range[0])
    stop = max(A_range[1], B_range[1])

    xs = np.linspace(start, stop, n_points)
    A_vals = cont_A['Prior'] * norm.pdf(xs, cont_A['Mean'], cont_A['Sigma'])
    B_vals = cont_B['Prior'] * norm.pdf(xs, cont_B['Mean'], cont_B['Sigma'])

    fig = plt.figure()
    colors = ['r', 'b']
    plt.plot(xs, A_vals, c=colors[0], label='A')
    plt.plot(xs, B_vals, c=colors[1], label='B')

    plt.axvline(x=q['t1'], c='k', lw=0.5, ls=':')
    plt.axvline(x=q['t2'], c='k', lw=0.5, ls=':')

    offset = 0.000007
    sub_level = -0.000025
    left = xs[0]
    right = xs[-1]

    def clip(x, lb, ub):
        res = x
        if res < lb:
            res = lb
        if res > ub:
            res = ub
        return res
    t1 = clip(q['t1'], xs[0], xs[-1])
    t2 = clip(q['t2'], xs[0], xs[-1])

    patch = patches.Rectangle([left, sub_level], t1-left, -sub_level-offset, angle=0.0,
                              color=colors[q['decision'][0]], alpha=0.75)
    plt.gca().add_patch(patch)
    patch = patches.Rectangle([t1, sub_level], t2-t1, -sub_level-offset, angle=0.0,
                              color=colors[q['decision'][1]], alpha=0.75)
    plt.gca().add_patch(patch)
    patch = patches.Rectangle([t2, sub_level], right-t2, -sub_level-offset, angle=0.0,
                              color=colors[q['decision'][2]], alpha=0.75)
    plt.gca().add_patch(patch)
    plt.legend()

    plt.title("Posterior probabilities and strategy q")
    plt.xlabel("image LR feature")
    plt.ylabel("posterior probabilities")


def compute_measurement_lr_discrete(imgs):
    """
    x = compute_measurement_lr_discrete(imgs)

    Calculates difference between left and right half of image(s).

    :param imgs:    set of images, (h, w, n) (or for color images (h, w, 3, n)) np.array
    :return:        (n, ) np.array of values in range <-10, 10>,
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


def montage(images, colormap='gray'):
    """
    Show images in grid.

    :param images:  np.array (h, w, n)
    """
    h, w, count = np.shape(images)
    h_sq = int(np.ceil(np.sqrt(count)))
    w_sq = h_sq
    im_matrix = np.zeros((h_sq * h, w_sq * w))

    image_id = 0
    for k in range(w_sq):
        for j in range(h_sq):
            if image_id >= count:
                break
            slice_w = j * h
            slice_h = k * w
            im_matrix[slice_h:slice_h + w, slice_w:slice_w + h] = images[:, :, image_id]
            image_id += 1
    plt.imshow(im_matrix, cmap=colormap)
    plt.axis('off')
    return im_matrix


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
