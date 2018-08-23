import autograd.numpy as np
from bcg.run_BCG import BCG
from bcg.model_init import Model

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

A, y = digits.data, digits.target
A = StandardScaler().fit_transform(A)

# classify small against large digits
y = (y > 4).astype(np.int)

A_train, A_test, y_train, y_test = train_test_split(
    A, y, train_size=1300, test_size=497, random_state=0)


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


# you can construct your own configuration dictionary
config_dictionary = {
        'solution_only': False,
        'verbosity': 'verbose',
        'dual_gap_acc': 1e-06,
        'runningTimeLimit': 40,
        'use_LPSep_oracle': True,
        'max_lsFW': 30,
        'strict_dropSteps': True,
        'max_stepsSub': 1000,
        'max_lsSub': 30,
        'LPsolver_timelimit': 100,
        'K': 1
        }


scale_parameter = 5
# define function evaluation oracle
def f(x):
    return np.sum([np.log(np.exp(-y_train[i]*np.dot(scale_parameter*A_train[i], x))+1) for i in range(len(A_train))])


res = BCG(f, None, l1Ball, config_dictionary)
print('optimal solution {}'.format(res[0]))
print('dual_bound {}'.format(res[1]))


def predict(data, label, weight):
    correct = 0
    for i in range(len(data)):
        y_est = 1/(1+np.exp(-np.dot(scale_parameter*data[i], weight)))
        if y_est > 0.5:
            y_est = 1
        else:
            y_est = 0
        if y_est == label[i]:
            correct += 1
    return correct/len(data)


acc_train = predict(A_train, y_train, res[0])
print('accuracy on the training dataset {}'.format(acc_train))

acc_test = predict(A_test, y_test, res[0])
print('accuracy on the testing dataset {}'.format(acc_test))

