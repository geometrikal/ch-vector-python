import numpy as np
import skimage.morphology as morph
import scipy.ndimage as sc
import pycircstat as pyc


def median_2d(th, radius=2):
    se = morph.disk(radius)
    return sc.generic_filter(th, pyc.median, footprint=se)


def delta_angle(alpha, beta):
    """
    Difference (alpha - beta) around the circle. Input angles do not need to
    be normalized in a given angle interval. Output is in (-pi,pi)
    """

    delta = np.angle(np.exp(1j * alpha) / np.exp(1j * beta))

    # for some values, e.g. alpha = beta = 0.49856791, the ratio of the complex numbers is very close to 0 but not zero...!
    if np.ndim(alpha) == 0 and np.ndim(beta) == 0:
        if alpha == beta:
            delta = 0.
    else:
        delta[alpha == beta] = 0.

    return delta


def circmedian(alpha):
    """
    Computes a median direction for circular data. alpha: 1-d array, in radians.
    Corrected version of pycircstat (simplified for 1D only), where result was depending
    on the order of the input and sometimes completely wrong in special cases
    (e.g. multiple solutions with even number of points). The algorithm is based
    on finding the diameter that splits the point distribution in two (50% of
    points on each side). The side of the diameter closest to the mean of the
    data is returned.

    Output in (-pi,pi)

    See https://hci.iwr.uni-heidelberg.de/sites/default/files/profiles/mstorath/files/storath2017fast.pdf
    for arc distance median vs bisecting median

    """

    # Checks
    if alpha.ndim > 1:
        raise Exception('circmedian only handles 1D arrays')

    # inits
    alpha = alpha.ravel()
    n = alpha.size
    is_odd = (n % 2 == 1)
    mean_alpha = circmean(alpha, low=-np.pi, high=np.pi, axis=0)  # return values in [-pi,pi]
    blnUseClosest = False  # use candidate closest to mean if True
    blnUseClosestPair = False  # use pair of balancing candidates closest to mean if True

    dd = delta_angle(alpha[:, np.newaxis],
                     alpha[np.newaxis, :])  # angular distance between all pair of points, in [-pi,pi]
    m1 = np.sum(dd >= 0, 0)  # For each point, number of points on one side of the point (itself included)
    m2 = np.sum(dd <= 0, 0)  # number of points on the other side of the point (itself included)
    dm = m1 - m2  # signed unbalance between sides
    dmabs = np.abs(dm)

    min_dm = np.min(dmabs)
    min_ind = np.argwhere(dmabs == min_dm).squeeze(axis=1)  # 1D array

    case = ''

    if is_odd:
        case += 'odd_'

        if min_dm > 0:
            # Example ???
            case += 'min>0_'
            print('NAN ', case, alpha)
            return np.nan  # hopeless

        if min_ind.size > 1:
            # multiple solutions
            # Example: [ 1.39079274, 0.17122657, -0.61367729, -2.56454636, 2.70582513]
            case += 'multiple_'
            blnUseClosest = True

    else:
        case += 'even_'

        if min_ind.size == 1:
            # Single min. Example???
            case += 'single_'
            if min_dm > 0:
                # NB: if == 0, the min is a solution.
                # Example ???
                case += 'min>0_'
                print('NAN ', case, alpha)
                return np.nan  # hopeless

        elif min_ind.size == 2 and np.sum(dm[min_ind]) != 0:
            # unpaired solution
            # Example ???
            case += '2minsUnpaired_'
            print('NAN ', case, alpha)
            return np.nan  # hopeless

        elif min_ind.size > 2:
            # multiple solutions
            case += 'multiple_'
            blnUseClosest = True

            if min_dm != 0:
                # Reduce candidates to pairs of balancing neighbours
                case += 'seekpairs_'
                pos_subinds = np.argwhere(dm[min_ind] > 0).squeeze(axis=1)  # ind of min_ind having positive unbalance
                pos_inds = min_ind[pos_subinds]  # ind of alpha having minimal positive unbalance
                blnUseClosestPair = True
                pairs = []
                for pos_ind in pos_inds:
                    deltas = dd[:, pos_ind]
                    # neighbour above
                    ind_above = np.nonzero(deltas > 0)[0]
                    ind_next_above = ind_above[np.argmin(deltas[ind_above])]  # index of alpha
                    if dm[ind_next_above] == -min_dm:
                        pairs.append([pos_ind, ind_next_above])
                    # neighbour below
                    ind_below = np.nonzero(deltas < 0)[0]
                    ind_next_below = ind_below[np.argmax(deltas[ind_below])]  # index of alpha
                    if dm[ind_next_below] == -min_dm:
                        pairs.append([pos_ind, ind_next_below])
                pairs_means = np.array([circmean(alpha[pair]) for pair in pairs])

                if not pairs:
                    # Example ???
                    case += 'min>0Unpaired_'
                    print('NAN ', case, alpha)
                    return np.nan
                elif len(pairs) == 1:
                    # E.g. [-0.43404185, -2.27418704, 0.45434327, -1.35741556, -0.32386174, -0.09667212, 0.90291071, -1.83059685, -1.78242314, 2.70070176]
                    case += 'singleValidPair_'
                    blnUseClosest = False
                    min_ind = np.array(pairs[0])
                # else: # multiple valid pairs. Example: [ 0, 1, 1, 2, 2, 3 ] (degenerate), [ 1, 1.2, 2, 2.2, 1+pi, 1.2+pi, 2+pi, 2.2+pi ] (ties)

    if blnUseClosest:
        # Take the closest value to the circular mean.
        phase_shift = [0, np.pi]  # account for +-pi degeneracy of the median
        ind_closest = []  # index of min_ind
        dist_closest = []

        for shift in phase_shift:

            if blnUseClosestPair:
                dist_to_mean = np.abs(delta_angle(pairs_means, mean_alpha + shift))
            else:
                dist_to_mean = np.abs(delta_angle(alpha[min_ind], mean_alpha + shift))

            ind_closest.append(np.argmin(dist_to_mean))
            dist_closest.append(dist_to_mean[ind_closest[-1]])

        if blnUseClosestPair:
            min_ind = np.array(pairs[ind_closest[np.argmin(dist_closest)]])
        else:
            min_ind_closest = min_ind[ind_closest[np.argmin(dist_closest)]]

            if is_odd:
                min_ind = np.array([min_ind_closest])

            else:
                # min_dm == 0 (otherwise blnUseClosestPair would be True)
                # the closest point is already a solution, although the number of
                # points is even. That means his partner point shares the same value (degenerate)
                # E.g. alpha = [ 0., 0., ..., 0. ]
                case += 'degenerate_'
                min_ind = np.array([min_ind_closest])

    if min_ind.size == 1:
        # check that we really have the median
        if dm[min_ind[0]] == 0:
            med = alpha[min_ind[0]]
        else:
            # Example ???
            print('NAN ', case, alpha)
            return np.nan
    else:
        # compute the median and check it
        med = pyc.mean(alpha[min_ind])
        dd_med = delta_angle(alpha, med)
        dm_med = np.sum(dd_med >= 0) - np.sum(dd_med <= 0)  # signed unbalance between sides
        if dm_med != 0:
            # Example: [ 3.8163336, -0.34886142, 2.09754686, 1.19611513, -3.72993505, -1.75901783, 4.69444219, 4.05781821, -4.41354647, -0.17710378]
            #          In that case, the used pair is opposite to the mean and wide apart, so that data points on the mean side are crossed when moving from one boundary to the other...
            case += 'circmeanNotMedian_'
            print('NAN ', case, alpha)
            return np.nan

    # Remove +-pi degeneracy of median by taking solution closest to mean
    if np.abs(delta_angle(med, mean_alpha)) > np.abs(delta_angle(med + np.pi, mean_alpha)):
        med += np.pi

    if case not in ['odd_', 'even_', 'odd_multiple_', 'even_multiple_seekpairs_singleValidPair_',
                    'even_multiple_seekpairs_', 'even_multiple_degenerate_']:
        print(case, alpha)

    return (med + np.pi) % (2. * np.pi) - np.pi  # convert to (-pi,pi). NB: % 2pi returns in [0,2pi]
