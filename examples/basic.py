
from bcg.run_BCG import BCG
from bcg.model_init import Model

import autograd.numpy as np


class Model_l1_ball(Model):
    def minimize(self, gradient_at_x=None):
        result = np.zeros(self.dimension)
        if gradient_at_x is None:
            result[0] = 1
        else:
            i = np.argmax(np.abs(gradient_at_x))
            result[i] = -1 if gradient_at_x[i] > 0 else 1
        return result

dimension = 100
l1Ball = Model_l1_ball(dimension)  # initialize the feasible region as a L1 ball of dimension 50000

# define function evaluation oracle and its gradient oracle
# the following example function is ||x||_2^2, where x is a n dimension vector

# CAREFUL: the gradient of the norm is not defined at 0!
#
# def f(x):
#     return np.linalg.norm(x, ord=2)**2
#
# this will not work with autograd! alternatively pass the correct gradient for the norm^2
#
# def f_grad(x):
#     return 2*x


def f(x):
    return np.dot(x,x)

res = BCG(f, None, l1Ball)
# res = BCG(f, f_grad, l1Ball)

print('optimal solution {}'.format(res[0]))
print('dual_bound {}'.format(res[1]))
print('primal value {}'.format(res[2]))