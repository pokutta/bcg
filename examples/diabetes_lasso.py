import autograd.numpy as np

from bcg.run_BCG import BCG

from bcg.model_init import Model


# load data
import pandas as pd
file_url = 'https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data'
diabetes = pd.read_csv(file_url, sep='\s+', header=None, skiprows=0)
diabetes = diabetes.rename(columns=diabetes.iloc[0]).drop(diabetes.index[0]).astype('float64')

A = np.array(diabetes.loc[:, diabetes.columns != 'Y'])
y = np.array(diabetes['Y'])
# split into training and testing data
A_train = A[:-40]
A_test = A[-40:]

y_train = y[:-40]
y_test = y[-40:]


# define a model class as feasible region
class Model_l1_ball(Model):
    def minimize(self, gradient_at_x=None):
        result = np.zeros(self.dimension)
        if gradient_at_x is None:
            result[0] = 1
        else:
            i = np.argmax(np.abs(gradient_at_x))
            result[i] = -1 if gradient_at_x[i] > 0 else 1
        return result


l1Ball = Model_l1_ball(A.shape[1])  # initialize the feasible region as a L1 ball

scale_parameter = 7

# define function evaluation oracle
def f(x):
    return np.dot(np.dot(scale_parameter*A_train, x) - y_train, np.dot(scale_parameter*A_train, x) - y_train)

# you can construct your own configuration dictionary
config_dictionary = {
        'solution_only': False,
        'verbosity': 'verbose',
        'dual_gap_acc': 1e-06,
        'runningTimeLimit': 15,
        'use_LPSep_oracle': True,
        'max_lsFW': 30,
        'strict_dropSteps': True,
        'max_stepsSub': 1000,
        'max_lsSub': 30,
        'LPsolver_timelimit': 100,
        'K': 1
        }

res = BCG(f, None, l1Ball, config_dictionary)
print('optimal solution {}'.format(res[0]))
print('dual_bound {}'.format(res[1]))

# prediction on train dataset
mse_train = res[2]/A_train.shape[0]
print('mean square error on training dataset {}'.format(mse_train))

# prediction on testing dataset
mse_test = np.dot(np.dot(scale_parameter*A_test, res[0]) - y_test, np.dot(scale_parameter*A_test, res[0]) - y_test)/A_test.shape[0]
print('mean square error on testing dataset {}'.format(mse_test))

