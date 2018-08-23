from bcg.model_init import Model
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh


class Model_simplex(Model):
    """LP model for the standard simplex.
    Dimension is one greater than that of the simplex itself.

    >>> simplex = Model_simplex(4)

    >>> simplex.minimize(np.array([2.0,-1.0,-0.3,.1]))
    array([ 0.,  1.,  0.,  0.])

    >>> utils.inSequence(simplex.augment(np.array([2.0,-1.0,-0.3,.1]),
    ...                                  np.array([0,0,0,1]), 0),
    ...                  [np.array([0,1,0,0]), np.array([0,0,1,0])])
    True
    """

    def minimize(self, gradient_at_x=None):
        result = np.zeros(self.dimension)
        result[np.argmin(gradient_at_x) if gradient_at_x is not None else 0] = 1
        return result


class Model_cube(Model):
    """LP model for cube.

    >>> cube = Model_cube(4)

    >>> cube.minimize(np.array([2.0,-1.0,-0.3,.1]))
    array([ 0.,  1.,  1.,  0.])

    >>> utils.inSequence(cube.augment(np.array([2.0,-1.0,-0.3,.1]),
    ...                               np.array([0,0,-1,0]), -0.7),
    ...                  [np.array([0,1,0,0]), np.array([0,1,1,0])])
    True
    """

    def minimize(self, gradient_at_x=None):
        assert gradient_at_x is None or len(gradient_at_x) == self.dimension
        if gradient_at_x is None:
            return np.zeros(self.dimension)
        else:
            return np.array(gradient_at_x < 0, dtype=float)
        
        
class Model_l1_ball(Model):
    """LP model for the ℓ1 ball.

    >>> l1Ball = Model_l1_ball(4)

    >>> l1Ball.minimize(np.array([2.0,-1.0,-0.3,.1]))
    array([-1.,  0.,  0.,  0.])

    >>> utils.inSequence(l1Ball.augment(np.array([2.0,-1.0,-0.3,.1]),
    ...                                 np.array([0,0,1,0]), -0.7),
    ...                  [np.array([-1,0,0,0]), np.array([0,1,0,0])])
    True
    """
    def minimize(self, gradient_at_x=None):
        result = np.zeros(self.dimension)
        if gradient_at_x is None:
            result[0] = 1
        else:
            i = np.argmax(np.abs(gradient_at_x))
            result[i] = -1 if gradient_at_x[i] > 0 else 1
        return result
    

class ModelBirkhoffPolytope(Model):
    """LP model of Birkhoff polytope.

    The convex hull of k-by-k permutation matrices.
    A matrix X is represented as a concatenation of its rows:

    [X[0, 0], X[0, 1], ..., X[0, k-1], X[1, 0], ..., X[k-1, k-1]]

    >>> m = ModelBirkhoffPolytope(3)
    >>> m.minimize(np.array([0, 1, 1, 1, 0, 1, 1, 1, 0]))
    array([ 1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.])

    The examples below are preceded with a cost matrix with an optimal
    dual solution to the left and above.  Entries marked with *...*
    are the 1-entries of the optimal solution, and /.../ mark other
    entries where the dual solution is tight.

    |   |   3 |   1 | 5    |
    |---+-----+-----+------|
    | 5 |   9 |   7 | *10* |
    | 3 |   7 | *5* | /8/  |
    | 1 | *4* | /2/ | /6/  |

    >>> m.minimize(np.array([9, 7, 10, 7, 5, 8, 4, 2, 6]))
    array([ 0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.])
    >>> np.inner(np.array([9, 7, 10, 7, 5, 8, 4, 2, 6]),
    ...          np.array([1, 0,  0, 0, 1, 0, 0, 0, 1]))
    20

    |   |   -1 |   0 |   1 |
    |---+------+-----+-----|
    | 0 | /-1/ | *0* | /1/ |
    | 1 |    1 | /1/ | *2* |
    | 2 |  *1* |   3 |   4 |

    >>> m.minimize(np.array([-1, 0, 1, 1, 1, 2, 1, 3, 4]))
    array([ 0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.])
    >>> np.inner(np.array([-1, 0, 1, 1, 1, 2, 1, 3, 4]),
    ...          np.array([0,  1, 0, 0, 0, 1, 1, 0, 0]))
    3
    """

    model = 0 # type: int
    """Size of the Birkhoff polytope: the number k when the polytope
    consists of k-by-k matrices.
    """

    def __init__(self, size: int):
        """Create LP model for Birkhoff polytope.

        Parameters:

        size: k for k-by-k matrices as elements of the polytope.
        """
        self.model = size
        super().__init__(dimension=size * size)

    def minimize(self, gradient_at_x=None):
        """Minimize over the Birkhoff polytope."""
        size = self.model
        if gradient_at_x is None:
            gradient_at_x = np.zeros(self.dimension)
        objective = gradient_at_x.reshape((size, size))
        matching = linear_sum_assignment(objective)
        solution = np.zeros((size, size))
        solution[matching] = 1
        return solution.reshape(self.dimension)


class ModelSpectrahedron(Model):
    """LP model of spectrahedron: {X psd | Tr[X] = 1}.

    X is represented as a vector by concatanating the truncated rows
    of its lower triangular part:
    [X[0, 0], X[1, 0], X[1, 1], ..., X[i, 0], ..., X[i, i-1],
    ..., X[k-1, k-1]].

    >>> m = ModelSpectrahedron(3)
    >>> m.minimize(np.array([2., 0., 1., 0., 0., 3.]))
    array([ 0.,  0.,  1.,  0.,  0.,  0.])

    In the following example:

    | eigenvector  | eigenvalue |
    |--------------+------------|
    | [.6, 0,  .8] |         -1 |
    | [.8, 0, -.6] |          2 |
    | [ 0, 1,   0] |          1 |


    Matrix:

    |  0.92 | 0. | -1.44 |
    |  0.   | 1. |  0.   |
    | -1.44 | 0. |  0.08 |

    Note that the non-diagonal entries should appear doubled in the
    objective:

    >>> np.linalg.norm(
    ...  m.minimize(np.array([0.92, 0.,  1., -2.88,  0.,  0.08]))
    ...  - np.array([0.36, 0., 0., 0.48, 0., 0.64]),
    ...  ord=float('inf')) < 10 ** -10
    True

    Test the value of objective function:

    >>> (np.inner(np.array([0.92, 0.,  1., -2.88,  0.,  0.08]),
    ...           np.array([0.36, 0., 0., 0.48, 0., 0.64]))
    ...  + 1.0 < 10 ** -10)
    True
    >>> (np.inner(np.array([0.92, 0.,  1., -2.88,  0.,  0.08]),
    ...           np.array([0.64, 0., 0., -0.48, 0., 0.36]))
    ...  - 2.0 < 10 ** -10)
    True
    """

    model = 0 # type: int
    """The size of the spectrahedron: k for the spectrahedron of
    k-by-k matrices.
    """

    def __init__(self, size: int):
        """Create a new LP model of spectrahedron.

        Parameters:

        size: k, where the spectrahedron consists of k-by-k matrices.
        """
        assert isinstance(size, int) and size >= 0, 'Not a nonnegative integer: %s' % size
        self.model = size
        dimension = size * (size + 1) // 2
        # Numpy advanced indices for converting between matrix and
        # vector form.  They trigger copying of numpy array.
        self.__matrix_to_vector = np.tril_indices(size)
        self.__vector_to_matrix = np.zeros((size, size), dtype=int)
        self.__vector_to_matrix[self.__matrix_to_vector] \
            = np.arange(dimension)
        self.__vector_to_matrix[self.__matrix_to_vector[::-1]] \
            = np.arange(dimension)
        super().__init__(dimension=dimension)

    def minimize(self, gradient_at_x=None):
        """Minimize linear function over the spectrahedron."""
        size = self.model
        if gradient_at_x is None:
            objective = np.zeros([size, size])
        else:
            objective = gradient_at_x[self.__vector_to_matrix]
            objective[np.eye(size) == 0] *= 0.5
        smallest_eigenvector = eigh(objective, eigvals=(0, 0))[1][:, 0]
        solution = (smallest_eigenvector[:, np.newaxis]
                    * smallest_eigenvector)
        return solution[self.__matrix_to_vector]