from math import inf
import os
import numpy as np
import datetime

from . import globs


################################
# line search functions
################################
def ternary_ls(obj_fct, x, direction, accuracy):
    gamma_ub = 1
    gamma_lb = 0
    # initialize
    y = x + direction  # end point
    endpoint_val = obj_fct.evaluate(y)
    val_y = endpoint_val
    val_x = obj_fct.evaluate(x)
    i = 0
    while abs(val_y - val_x) > accuracy:
        zx = x + 1/float(3) * (y - x)
        zy = x + 2/float(3) * (y - x)
        value_zx = obj_fct.evaluate(zx)
        value_zy = obj_fct.evaluate(zy)
        if value_zx < value_zy:
            y = zy
            gamma_ub = gamma_lb + (gamma_ub-gamma_lb) * 2/3
            val_y = value_zy  # update value y because position of y changed
        else:
            x = zx
            gamma_lb = gamma_lb + (gamma_ub-gamma_lb) * 1/3
            val_x = value_zx  # update value x because position of x changed
        i += 1
    return gamma_lb, i


def backtracking_ls_FW(objectiveFunction, x, grad, direction, steps):
    step_size = 1
    grad_direction = np.inner(grad, direction)

    i = 0
    # assert grad_direction <= 0, 'grad_direction is {}'.format(grad_direction)
    if grad_direction == 0:
        return 0, i

    evalu_oldpint = objectiveFunction.evaluate(x)
    evalu_newpoint = objectiveFunction.evaluate(x + step_size * direction)
    while (evalu_newpoint - evalu_oldpint) > globs.ls_eps * step_size * grad_direction:
        if i > steps:
            if evalu_oldpint - evalu_newpoint >= 0:
                return step_size, i
            else:
                return 0, i
        step_size *= globs.ls_tau
        evalu_newpoint = objectiveFunction.evaluate(x + step_size * direction)
        i += 1
    # assert (evalu_oldpint - evalu_newpoint >= 0)
    return step_size, i


def backtracking_ls_on_alpha(alpha_list, objectiveFunction, s_list, step_size_ub, direction, steps,
                             func_val_improve_last, strict_dropSteps = True):
    """
    backtracking line search method from https://people.maths.ox.ac.uk/hauser/hauser_lecture2.pdf
    used on sub-algorithm
    """

    step_size = step_size_ub
    grad_direction = -np.inner(direction, direction)

    x_old = np.dot(np.transpose(s_list), alpha_list)
    x_new = np.dot(np.transpose(s_list), alpha_list + step_size * direction)  # end point
    evalu_oldpint = objectiveFunction.evaluate(x_old)
    evalu_newpoint = objectiveFunction.evaluate(x_new)

    # relax dropping criterion
    if func_val_improve_last != 'N/A':
        if not strict_dropSteps:
            drop_criteria = min(0.5 * func_val_improve_last, globs.ls_eps)
        else:
            drop_criteria = 0
        if evalu_newpoint <= evalu_oldpint + drop_criteria:
            return step_size, 0, 'P'

    # begin line search
    i = 0
    while (evalu_newpoint - evalu_oldpint) > globs.ls_eps*step_size * grad_direction:
        if i > steps and evalu_newpoint - evalu_oldpint >= 0:
            return 0, i, 'PS'
        step_size *= globs.ls_tau
        x_new = np.dot(np.transpose(s_list), alpha_list + step_size * direction)
        evalu_newpoint = objectiveFunction.evaluate(x_new)
        i += 1
    if evalu_newpoint >= evalu_oldpint:
        return 0, i, 'PS'
    return step_size, i, 'P'


################################
# cache functions:
################################
def inSequence(array, sequence):
    """Return True when Numpy array is an element of sequence.

    >>> inSequence(np.array([1,2,3]), [np.array([0,1,2]),
    ...                                np.array([1.0, 2.0, 3.0])])
    True

    >>> inSequence(np.array([1,2,3]), [np.array([0,1,2]),
    ...                                np.array([-2.0, 1.0, 3.0])])
    False
    """
    for i in sequence:
        if np.all(array == i):
            return True
    return False


def removeFromCache(x):
    """Remove point x from cache if there.
    >>> _ignore = reset_cache()
    >>> for i in range(3):
    ...     _ignore = addToCache(np.array([i]))
    >>> removeFromCache(np.array([2]))
    point deleted from cache, current number of points in cache 2
    >>> removeFromCache(np.array([3]))

    >>> removeFromCache(np.array([1]))
    point deleted from cache, current number of points in cache 1
    """
    current_cache_length = len(globs.previousPoints)
    key = hash(x.tostring())
    try:
        del globs.previousPoints[key]
    except KeyError:
        pass
    else:
        assert current_cache_length - len(globs.previousPoints) == 1


def addToCache(x, clean=None):
    if clean:
        result = dict(globs.previousPoints)
        current_value = np.inner(x, x)
        for key, y in globs.previousPoints.items():
            if np.inner(x, y) > current_value:
                result.pop(key)
        globs.previousPoints = result
    key = hash(x.tostring())
    if key not in globs.previousPoints:
        globs.previousPoints[key] = x


def checkForCache(c, goal):
    """Search for a cached numpy array with small objective value.

    c: objective
    goal: upper bound on the acceptable objective value.

    >>> reset_cache()

    >>> _ignore = addToCache(np.array([1., 0.]))

    >>> _ignore = addToCache(np.array([0., 1.]))

    >>> checkForCache(np.array([1,2]), goal=1)
    array([ 1.,  0.])

    >>> checkForCache(np.array([2,1]), goal=1)
    array([ 0.,  1.])

    >>> checkForCache(np.array([1,3]), goal=.5)

    """
    for x in globs.previousPoints.values():
        if np.inner(c, x) <= goal:
            break
    else:
        x = None
    return x


def checkForPairwiseCache(c, c_tilde, goal):
    mi = inf
    x_plus = None
    mi_tilde = inf
    x_minus = None

    for x in globs.previousPoints.values():
        if np.inner(c, x) < mi:
            mi = np.inner(c, x)
            x_plus = x
        if np.inner(c_tilde, x) < mi_tilde:
            mi_tilde = np.inner(c_tilde, x)
            x_minus = x
        if mi + mi_tilde <= goal:
            break
    return x_plus, x_minus


def find_closest_cache(c):
    m = inf
    m_x = None
    for x in globs.previousPoints.values():
        if np.inner(c, x) < m:
            m_x = x
            m = np.inner(c, x)
    return m_x


def reset_cache():
    # reset global statistic variables
    globs.previousPoints = {}


####################################
# for reporting on console
####################################
def console_header(all_headers, run_config):
    # under normal mode
    header = all_headers[:3] + all_headers[4:6] + [all_headers[7]]
    width = np.array([12, 8, 22, 22, 12, 12])
    # it will be: ['Iteration', 'Type', 'Function Value', 'Dual Bound', '#Atoms', 'WTime']
    if run_config['verbosity'] == 'verbose':
        header += [all_headers[3]]  # add 'Primal Improve' to the console output
        width = np.append(width, 22)
    return header, width


def console_info(all_info, run_config):
    # under normal mode
    info = all_info[:3] + all_info[4:6] + [all_info[7]]
    if run_config['verbosity'] == 'verbose':
        info += [all_info[3]]  # add 'Primal Improve' to the console output
    return info

