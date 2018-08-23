# contains oracles (LMO and Weak Separation)
# contains feasible region construction (unit cube, L1Ball, Simple, or from lp file, Birkhoffpolytope, spectrahedron)

import abc
import numbers
import logging
import numpy as np

from . import utils
from . import globs


class Model(abc.ABC):
    """A generic class for an LP problem.

    Attributes:

    dimension: The number of coordinates feasible solutions have.


    Implementation specific attributes:
    These may be missing.

    model: Dynamic data for optimizing over the model.
    """

    def __init__(self, dimension):
        """Initialize a model."""
        assert (isinstance(dimension, numbers.Integral)
                and dimension >= 0)
        super().__init__()
        self.dimension = dimension

    @abc.abstractmethod
    def minimize(self, cc=None):
        """Minimize objective cc."""
        pass

    def augment(self, cc=None, x=None, goal=None):
        # for models without usage of Gurobi: no early termination, return self.minimize(cc)
        # for models with usage of Gurobi: this method will be redefined in LPsolver.py
        """Find a solution smaller than value goal for objective cc.
        An already known solution is x if not None."""
        return self.minimize(cc)

    def solve(self, cc=None):  # Linear Optimization Oracle
        return self.minimize(cc)

    def weak_sep(self, cc=None, x=None, strict=True, extra_margin=0):  # Weak Separation Oracle
        """Find a solution for cc with value smaller than that of x.
            The value should be smaller by at least extra_margin."""

        if cc is not None and x is not None:
            goal = np.inner(cc, x) - extra_margin
            if strict:
                goal -= globs.accuracyComparison
            else:
                goal += globs.accuracyComparison
            if globs.useCache:
                y = utils.checkForCache(cc, goal)
                if y is not None:
                    # logging.info('---> found cached point with <c, y> value {}'.format(np.inner(cc, y)))
                    return y, 'FIC'
        else:
            logging.info('finding first feasible point ...... ')

        s = self.augment(cc, x, goal)

        if True:
            if globs.useCache:
                utils.addToCache(s)

            # logging.info('condition: {}'.format(np.inner(cc, s) - goal))
            if x is None or cc is None or (np.inner(cc, s) < goal):
                # logging.info('found an improving vertex!')
                return s, 'FI'
            else:
                # logging.info('Did not find better s!')
                # logging.info('function value:{}'.format(np.inner(cc, s)))
                # logging.info('old value:{}'.format(np.inner(cc, x)))
                if globs.useCache:  # add this point to cache
                    s = utils.find_closest_cache(cc)
                return s, 'FN'







