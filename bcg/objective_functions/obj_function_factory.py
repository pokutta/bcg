"""Create objective function by name.

Currently, the name is the name of a class from the module with almost
the same name.  The module name is obtained from name by converting
all capital letters to lower case and inserting a _ before them.
Exceptions are capital letters following a _, which are left alone.
All leading _ are stripped from the module name.

Examples:

>>> import numpy as np

>>> f = ObjectiveFunctionFactory.create_function(
...        'standard_parabola', np.arange(3))
>>> float(round(f.evaluate(np.array([1, 0, -1])), 5))
11.0

>>> ObjectiveFunctionFactory.create_function(
...     'square_norm_Ax_minus_b', 4, 3, 2)
... # doctest: +ELLIPSIS
<objective_functions.square_norm_Ax_minus_b.square_norm_Ax_minus_b object at ...>
>>> ObjectiveFunctionFactory.create_function(
...     'lasso_function', 5)
... # doctest: +ELLIPSIS
<objective_functions.lasso_function.lasso_function object at ...>
>>> ObjectiveFunctionFactory.create_function(
...     'sdp_matrixCompletion', size=5, density=1/3,
...     rank_ratio=2/5)
... # doctest: +ELLIPSIS
<objective_functions.sdp_matrix_completion.sdp_matrixCompletion object at ...>
>>> ObjectiveFunctionFactory.create_function(
...     'MatrixCompletion', np.arange(6).reshape(3, 2), 0.1)
... # doctest: +ELLIPSIS
<objective_functions.matrix_completion.MatrixCompletion object at ...>
"""
# TODO: Test the created functions.  Unfortunately, most of them is random, so hard to check for specific values.

import importlib
import re


class ObjectiveFunctionFactory(object):

    @staticmethod
    def create_function(name, *args, **kwargs):
        def module_name(match):
            return '_' + match.group().lower()

        module_name = re.sub(r'(?<!_)[A-Z]', module_name, name) \
                        .lstrip('_')
        module = importlib.import_module('objective_functions.'
                                         + module_name)
        obj_function = getattr(module, name)
        return obj_function(*args, **kwargs)
