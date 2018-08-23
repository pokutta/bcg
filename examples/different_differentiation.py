# DEMONSTRATES THE USE OF AUTOMATIC DIFFERENTIATION VIA autograd

from bcg.run_BCG import BCG
from bcg.model_init import Model
from bcg import globs

import importlib.util
spec = importlib.util.find_spec("autograd")
if spec is None:
    print("No Autograd.")
    import numpy as np
else:
    print("Autograd detected.")
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
# the following example function is (x-shift)^2, where x is a n dimension vector
shift = np.random.randn(dimension)

def f(x):
    return np.linalg.norm(x - shift, ord=2)**2

def f_grad(x):
    return 2*(x - shift)


# DEMONSTRATE AUTOMATIC DIFFERENTIATION
# NOTE: if you want to use autograd you have to use the autograd.numpy wrapper already for the function definition
# see above

# gradient provided
print("\n\n")
print("Using provided gradient.")
res = BCG(f, f_grad, l1Ball)
print('optimal solution {}'.format(res[0]))
print('dual_bound {}'.format(res[1]))
print('primal value {}'.format(res[2]))

# autograd gradient
print("\n\n")
print("Using autograd.")
res = BCG(f, None, l1Ball)
print('optimal solution {}'.format(res[0]))
print('dual_bound {}'.format(res[1]))

# numerical gradient
print("\n\n")
print("Using numerical gradient approximation.")
# ignore autograd package
# only for demonstration and not recommended
globs.autograd = False

res = BCG(f, None, l1Ball)
print('optimal solution {}'.format(res[0]))
print('dual_bound {}'.format(res[1]))