import abc


class ObjFunction(abc.ABC):
    """Objective function with gradient.

    >>> import numpy as np
    >>> class TestFunction(ObjFunction):
    ...
    ...     def evaluate(self, value):
    ...         return value[0] ** 2 - value[1] ** 3
    ...
    ...     def gradient(self, value):
    ...         return np.array([2 * value[0], -3 * value[1] ** 2])
    >>> t = TestFunction()
    >>> t.evaluate(np.array([1, -1]))
    2
    >>> t.gradient(np.array([1, -1]))
    array([ 2, -3])
    """
    @abc.abstractmethod
    def evaluate(self, value):
        """Return function value at point value."""
        return

    @abc.abstractmethod
    def gradient(self, value):
        """Return gradient at point value.

        >>> import numpy as np
        >>> class TestFunction(ObjFunction):
        ...
        ...     def evaluate(self, value):
        ...         return np.sum(value * value)
        ...
        ...     def gradient(self, value):
        ...         return super().gradient(value)
        >>> np.around(
        ...     TestFunction().gradient(np.arange(6).reshape(2, 3)),
        ...     5)
        array([[  0.,   2.,   4.],
               [  6.,   8.,  10.]])


        Implementation notes:

        scipy.optimize.approx_fprime doesn't support functions with
        multidimensional array argument:

        >>> import numpy as np
        >>> from scipy.optimize import approx_fprime
        >>> def x_x_transpose(x):
        ...     return np.sum(x * x)
        >>> np.around(approx_fprime(np.arange(6),
        ...                         x_x_transpose,
        ...                         np.sqrt(np.finfo(float).eps)),
        ...           5)
        array([  0.,   2.,   4.,   6.,   8.,  10.])
        >>> np.around(approx_fprime(np.arange(6).reshape(2, 3),
        ...                         x_x_transpose,
        ...                         np.sqrt(np.finfo(float).eps)),
        ...           5) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        ValueError: operands could not be broadcast together with shapes (2,3) (2,)

        Correct output should be:

        array([[  0.,   2.,   4.],
               [  6.,   8.,  10.]])
        """
        from scipy.optimize import approx_fprime
        import numpy as np
        shape = value.shape

        def f(x):
            return self.evaluate(x.reshape(shape))

        gradient = approx_fprime(value.reshape(-1),
                                 f,
                                 np.sqrt(np.finfo(float) .eps))
        return gradient.reshape(shape)

    def check_gradient(self, value, accuracy=None):
        """Verify gradient of function.

        Return True if gradient is correct, otherwise the difference
        in two norm.

        Parameters:

        value: the point at which to verify gradient

        accuracy: allowed error in gradient in two norm, due to
                  numerical errors and gradient estimation.

        Examples:

        >>> import numpy as np
        >>> def f(x):
        ...     return np.sum(x ** 3)
        ...
        >>> def f_grad(x):
        ...     return 3 * x ** 2
        ...
        >>> def f_grad_wrong_shape(x):
        ...     # Flattening array is an error
        ...     return f_grad(x).reshape(-1)
        ...
        >>> def f_wrong_grad(x):
        ...     return 2 * x ** 2
        ...
        >>> test = ObjectiveFunction(f, f_grad)
        >>> test_wrong_shape = ObjectiveFunction(f, f_grad_wrong_shape)
        >>> test_wrong_grad = ObjectiveFunction(f, f_wrong_grad)

        >>> test.check_gradient(np.random.randint(10, size=(2, 3)),
        ...                     10 ** -5)
        True
        >>> test.check_gradient(np.array([[2, 0, 1], [-1, 0, -3]]))
        True
        >>> test_wrong_shape.check_gradient(np.array([[2, 1, 1],
        ...                                           [-1, 0, -3]]))
        Traceback (most recent call last):
            ...
        ValueError: Gradient shape is (6,) instead of (2, 3)


        The float() conversion below is a workaround for type
        np.float64 not providing the shortest repersentation.

        >>> float(round(test_wrong_grad.check_gradient(
        ...     np.array([[2, 0, 1], [-1, 0, -3]])), 5))
        9.94987
        """
        from scipy.optimize import check_grad, approx_fprime
        import numpy as np
        shape = value.shape

        def f(x):
            return self.evaluate(x.reshape(shape))

        if accuracy is None:
            estimated_gradient = approx_fprime(value.reshape(-1), f,
                                    np.sqrt(np.finfo(float).eps))
            accuracy = (np.linalg.norm(estimated_gradient)
                        * np.cbrt(np.finfo(float).eps))

        def grad(x):
            gradient = self.gradient(x.reshape(shape))
            if gradient.shape != shape:
                raise ValueError('Gradient shape is %s instead of %s'
                                 % (gradient.shape, shape))
            return gradient.reshape(-1)

        error = check_grad(f, grad, value.reshape(-1))
        return True if error <= accuracy else error


class ObjectiveFunction(ObjFunction):
    """Explicitly given objective function.

    Example:

    >>> import numpy as np
    >>> def f(x):
    ...     return np.sum(x ** 3)
    ...
    >>> def f_grad(x):
    ...     return 3 * x ** 2
    ...
    >>> test = ObjectiveFunction(f)
    >>> test2 = ObjectiveFunction(f, f_grad)

    >>> test.evaluate(np.array([2, 1]))
    9
    >>> test2.evaluate(np.array([2, 1]))
    9

    >>> np.around(test.gradient(np.array([2, 1])), 5)
    array([ 12.,   3.])
    >>> test2.gradient(np.array([2, 1]))
    array([12,  3])

    >>> test.evaluate(np.array([[2, 1], [-1, -3]]))
    -19
    >>> test2.evaluate(np.array([[2, 1], [-1, -3]]))
    -19

    >>> np.around(test.gradient(np.array([[2, 1], [-1, -3]])), 5)
    array([[ 12.,   3.],
           [  3.,  27.]])
    >>> test2.gradient(np.array([[2, 1], [-1, -3]]))
    array([[12,  3],
           [ 3, 27]])
    """
    def __init__(self, function, gradient=None):
        """Wrap a function into an instance of ``ObjFunction``.

        Parameters:

        function: The objective function as a callable.
        gradient: The gradient of ``function`` as a callable.
                  If omitted the gradient will be estimated.
        """
        self.__function = function
        if gradient is not None:
            self.__gradient = gradient
        else:
            self.__gradient = super().gradient

    def evaluate(self, value):
        return self.__function(value)

    def gradient(self, value):
        return self.__gradient(value)


if __name__ == '__main__':
    import numpy as np

    # version 1
    def f1(x):
        return np.sum(x ** 2)

    def f1_grad(x):
        return 2 * x
    objfunction1 = ObjectiveFunction(f1, f1_grad)
    print(objfunction1.check_gradient(np.array([2, 1, 1])))  # when input is a vector, correct
    print(objfunction1.check_gradient(np.array([[2, 1, 1], [-1, 0, -3]])))  # when input is a matrix, correct

    # version 2
    def f2(x):
        return np.linalg.norm(x, ord=2) ** 2

    def f2_grad(x):
        res = 2*x
        return res

    objfunction2 = ObjectiveFunction(f2, f2_grad)
    print(objfunction2.check_gradient(np.array([2, 1, 1])))  # when input is a vector
    print(objfunction2.check_gradient(np.array([[2, 1, 1], [-1, 0, -3]])))  # when input is a matrix, it's incorrect
    # explanation:
    print(f2(np.array([[2, 1, 1], [-1, 0, -3]])))
    # 2 norm of matrix A is the square root of the maximum eigenvalue of A^(H)A
