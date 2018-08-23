import numpy as np
from bcg.run_BCG import BCG


# define function evaluation oracle and its gradient oracle
def f(x):
    return np.linalg.norm(x, ord=2)**2


def f_grad(x):
    return 2*x


res = BCG(f, f_grad, 'spt10.lp')
print('optimal solution {}'.format(res[0]))
print('dual_bound {}'.format(res[1]))


# you can also construct your own configuration dictionary. For example, let's set the 'solution_only' to be False here.
config_dictionary = {
        'solution_only': False,
        'verbosity': 'verbose',
        'dual_gap_acc': 1e-06,
        'runningTimeLimit': 2,
        'use_LPSep_oracle': True,
        'max_lsFW': 30,
        'strict_dropSteps': True,
        'max_stepsSub': 200,
        'max_lsSub': 30,
        'LPsolver_timelimit': 100,
        }
res = BCG(f, f_grad, 'spt10.lp', run_config=config_dictionary)
print('optimal solution {}'.format(res[0]))
print('dual_bound {}'.format(res[1]))
print('primal value {}'.format(res[2]))



