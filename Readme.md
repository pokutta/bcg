# Blended Conditional Gradient -- Frank-Wolfe algorithm enhanced with gradient descent

This package implements an algorithm for minimizing a smooth convex function over a polytope P, that combines the Frank-Wolfe algorithm (also called conditional gradient) with gradient-based steps. Different from the pairwise-step and away-step variants of Frank-Wolfe algorithm, it makes better use of the active vertices via gradient steps with a guarantee of significant progress.

For more information on this algorithm, please refer to the paper [Blended Conditional Gradients: the unconditioning of conditional gradients](https://arxiv.org/abs/1805.07311)

## Prerequisites
* Optional: LP solver (example: Gurobi)
  * LP oracles can also be directly defined not relying on Gurobi etc.
* package `autograd` (optional): simplifies sepcifying gradients.

## Installation

Standard method:

```sh
git clone https://github.com/pokutta/bcg.git
pip install .
```

Source code is available at <https://github.com/pokutta/BCG>.

## Changelog

Version 0.1.0: *initial release*

## Known Limitations

Many... but we are working on them. Most importantly:

1. Currently the code is not thread safe \
   (considered critical and will be addressed in one of the next versions)


## Get Started
```BCG``` is the only function you may need to call to use this package.The following is a detailed explanation on its input parameters and return value.

Syntax: `BCG(f, f_grad, model, run_config=None)`
* **Parameters**
    * ``f``: callable func(x). \
    Objective function. It should be smooth convex.
    * ``f_grad``: callable f_grad(x), optional. \
    Gradient of the objective function. If set to None, gradient will
    be determined automatically, using package `autograd` if
    available, otherwise a possibly slow approximation will be used.
    * ``model``: {filename, Class}, used for constructing feasible region. \
    BCG function accepts the following two arguments as the source of feasible region:
        1. LP file (Gurobi will be used as LP solver by default. Make changes in LPsolver.py if you want to use other LP solvers)
        2. user-defined model class variable(LP Oracle at the gradient vector should be defined inside)
    * ``run_config``: a dictionary, optional  \
    If not provided, the following default setting will be used：
        ```python
        run_config = {
            'solution_only': True,
            'verbosity': 'normal',
            'dual_gap_acc': 1e-06,
            'runningTimeLimit': None,
            'use_LPSep_oracle': True,
            'max_lsFW': 30,
            'strict_dropSteps': True,
            'max_stepsSub': 200,
            'max_lsSub': 30,
            'LPsolver_timelimit': 100,
            'K': 1
        }
        ```
        Explanation of the above configuration keys can be found [here](#configuration-dictionary).
* **Returns:**  3-d list \
    First element is the optimal point (a n-dimension vector) and the second element is the dual bound when the algorithm ends, and the last element is the primal function value on current optimal point.\
    The logging file and a dictionary containing algorithm performance information will also be saved if run_configuration['solution_only'] is set to False.




## Example Usage 1: with a LP file as feasible region
```python
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
print('primal value {}'.format(res[2]))
```
You can also construct your own configuration dictionary. For example, let's set the 'solution_only' to be False here.
```python
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
    'K': 1.1
}
         
res = BCG(f, f_grad, 'spt10.lp', run_config=config_dictionary)
 
print('optimal solution {}'.format(res[0]))
print('dual_bound {}'.format(res[1]))
print('primal value {}'.format(res[2]))
```

## Example Usage 2: with a user-defined model class as feasible region
```python
import numpy as np
from bcg.run_BCG import BCG

# define a model class to construct feasible region
# the following example is L1 ball, other examples can be found at example_model_class.py
from bcg.model_init import Model

class Model_l1_ball(Model):
    def minimize(self, gradient_at_x=None):
        result = np.zeros(self.dimension)
        if gradient_at_x is None:
            result[0] = 1
        else:
            i = np.argmax(np.abs(gradient_at_x))
            result[i] = -1 if gradient_at_x[i] > 0 else 1
        return result

l1Ball = Model_l1_ball(100)  # initialize the feasible region as a L1 ball of dimension 100

# define function evaluation oracle and its gradient oracle
# the following example function is (x-shift)^2, where x is a n dimension vector
shift = np.random.randn(100)
def f(x):
    return np.linalg.norm(x - shift, ord=2)**2

def f_grad(x):
    return 2*(x - shift)


res = BCG(f, f_grad, l1Ball)
print('optimal solution {}'.format(res[0]))
print('dual_bound {}'.format(res[1]))
print('primal value {}'.format(res[2]))
```
## Output Example
```
Using provided gradient.
dimension of this feasible_region is 100
checking validity of function gradient...
╭────────────────────────┬────────────────────────┬────────────────────────┬────────────────────────┬────────────────────────┬────────────────────────╮
│       Iteration        │          Type          │     Function Value     │       Dual Bound       │         #Atoms         │         WTime          │
├────────────────────────┼────────────────────────┼────────────────────────┼────────────────────────┼────────────────────────┼────────────────────────┤
│                      1 │                     FI │      91.52256902109498 │      2.945672332562157 │                      1 │                 0.0030 │
│                      2 │                     FN │      91.21266631260148 │      2.945672332562157 │                      2 │                 0.0041 │
│                      3 │                     FI │      91.12222718958402 │     0.7872772175737643 │                      3 │                 0.0099 │
│                      4 │                     FN │      91.09363112732508 │     0.7872772175737643 │                      4 │                 0.0115 │
│                      5 │                     FI │      91.07786291882721 │    0.19708572371482436 │                      5 │                 0.0125 │
│                      6 │                      P │      91.07469121674026 │    0.19708572371482436 │                      5 │                 0.0133 │
│                      7 │                     FN │      91.07380013379127 │    0.19708572371482436 │                      5 │                 0.0145 │
│                      8 │                      P │       91.0733404209301 │   0.028801617894984144 │                      5 │                 0.0151 │
│                      9 │                      P │      91.07321459940893 │   0.028801617894984144 │                      5 │                 0.0158 │
│                     10 │                     FN │      91.07319272605285 │   0.028801617894984144 │                      5 │                 0.0169 │
│                     11 │                      P │      91.07318511979354 │   0.004507642594342887 │                      5 │                 0.0175 │
│                     12 │                      P │       91.0731824553222 │   0.004507642594342887 │                      5 │                 0.0184 │
│                     13 │                     FN │      91.07318194664359 │   0.004507642594342887 │                      5 │                 0.0195 │
│                     14 │                      P │      91.07318186214543 │  0.0007185411138372899 │                      5 │                 0.0201 │
│                     15 │                     FN │      91.07318186214543 │  0.0007185411138372899 │                      5 │                 0.0213 │
│                     16 │                      P │      91.07318186214513 │  6.007732255008946e-07 │                      5 │                 0.0220 │
Achieved required dual gap accuracy, save results and exit BCG algorithm
╰────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────╯
optimal solution [ 0.          0.          0.40293517  0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
 -0.11816443  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
 -0.12102891  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.16765871  0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.19021278]
dual_bound 6.007732255008946e-07
primal value 91.07318186214513
```
## Configuration dictionary
* `solution_only`: binary \
If set to True, output will be the optimal solution and dual bound when the algorithm ends. \
If set to False, besides returning optimal solution and dual bound, a dictionary with keys 'func_val_all', 'wallClock_all', and 'dual_bound_all' will be saved as a pickle file in current working directory. Each key is associated with an array that contains per-iteration information on function value, wall-clock time and dual bound. Additionally, the logging file will also be saved in current working directory in this case. The logging file will be presented as a table with the following headers:
  * `Iteration`: iteration count for current iteration.
  * `Type`: A string which stands for the type of current iteration. A Frank-Wolfe iteration will start with `F` while a gradient iteration (sometimes we may also call it a sub-iteration) will start with `P`. When weak separation oracle is used, `FIC` means improving vertex found in cache, `FI` means improving vertex found but not in cache, and `FN` means no improving vertex found. If any vertex is dropped during this iteration, the string will end with `D`. And ending with `S` means line search gets stuck during current iteration.
  * `Function Value`: function value in current iteration.
  * `Primal Improve`: improvement on function value compared to last iteration, calculated as $f(t-1)-f(t)$
  * `Dual Bound`: If regular linear optimization oracle is used, this columns shows the Frank-wolfe gap in current iteration. Otherwise, when weak separation oracle is used, this value will be an approximation of Frank-wolfe gap.
  * `#Atoms`: number of active atoms.
  * `Iteration Time`: wall clock time spent in current iteration
  * `WTime`: accumulated wall clock time until current iteration.
  * `step size`: step size in current iteration.
  *  `num_ls`: number of steps taken to find the near-optimal step size in the line search.
* `verbosity`: {'normal', 'verbose', 'quiet'} \
This option allows you different output modes on the console while running the algorithm.
  * `normal`: show default table, including 'Iteration', 'Type', 'Function Value', 'Primal Improve', 'Dual Bound', '#Atoms', and 'WTime'.
  * `verbose`: show extended table, which adds the 'Primal Improve' column compared to the normal mode.
  * `quiet`: no table will be shown on the console
* `dual_gap_acc`: float \
This value will be used as the default stop criterion. The algorithm stops when dual gap is smaller than this value.
* `runningTimeLimit`: {int, None}  \
CPU time limit for running the algorithm, measured in seconds. The algorithm stops when reaching time limit or when dual bound smaller than the above 'dual_gap_acc' value, whichever comes first. \
If set to ```None```, algorithm stops only according to 'dual_gap_acc' value.
* `use_LPSep_oracle`: binary  \
If set to True, the weak separation oracle will be used. Provided with a point *x* and its gradient of the objective *c*, this oracle decides whether there exists *s* with *(cs − cx)* larger than current dual bound in feasible region.  \
If set to False, use a regular LP oracle that returns a point where its inner product with current gradient is minimized. \
The weak separation oracle is an accelerated substitute of the LP oracle traditionally used in the Frank–Wolfe procedure, providing a good enough, but possibly far from the optimal solution to an LP problem. You may want to choose the weak separation oracle when calling the LP oracle every iteration is computationally expensive.
* `strict_dropSteps`: binary \
If set to True, a vertex will be dropped if function value at this vertex is strictly smaller than current function value. \
If set to False, a vertex will be dropped as long as function value at this vertex is smaller than current function value plus half of the function value improvement the algorithm had made in the last iteration. The relaxed dropping criterion here allows a controlled increase of the objective function value in return for additional sparsity. Therefore, you may want to set it to False when a more sparse solution is desired.
* `max_lsFW`: int \
Maximum number of line search steps per Frank-Wolfe iteration.
* `max_stepsSub`: int \
Maximum number of subroutine iterations.
* `max_lsSub`: int \
Maximum number of line search steps per subroutine iteration.
* `LPsolver_timelimit`: {int, float}, could be omitted if no LP solver is used. \
Running time limit on LP solver per iteration, measured in seconds
* `K` : float \
Accuracy parameter used to scale target dual bound value in the weak separation oracle and also in the entering criterion of SiGD iteration.

## What is shown on console?
* `performance table` \
Display of the performance table depends on the value of `verbosity` in the run configuration.
* `exit code`: \
Exit code of BCG. Usually the algorithm stops when reaching `dual_gap_acc` or `runningTimeLimit`, but the algorithm will also stops due to the following reasons.
  * `0`: The algorithm stopped because of keyboard interruption
  * `1`: The algorithm stopped because no further primal progress has been detected, relative to the line search accuracies $2^{-\lambda}$, where $\lambda$ is either `max_lsFW` (for FW steps) and `max_lsSub` (for GD steps). \
  This usually happens when the dual bound is already very small (but not yet below its threshold) and the derived primal progress of the order of $\operatorname{dual\_bound}^2 / C$, where $C$ is the curvature constant is too small.
  * `2`: The algorithm stopped because it reached the dual progress bound `dual_gap_acc`.
  * `3`: The algorithm stopped because it reached the time limit `runningTimeLimit`.

## Using automatic differentiation with Autograd
If ```f_grad=None``` and the ```autograd```package is installed, BCG will use ```autograd``` to attempt automatic differentation. If this fails it will use ```scipy.approx_fprime```'s finite-difference approximation for computing numerical gradients; it will use the numerical gradients also if ```autograd``` is not present or if deactivated via ```globs.autograd = False```. For ```conda``` users, the ```autograd``` package can be installed with ```conda install -c omnia autograd```. Note that the ```autograd``` numpy wrapper has to be used for the function definition in order to work with ```autograd```. We provide an example now:

```python
import autograd.numpy as np
from bcg.run_BCG import BCG

# define a model class as feasible region
# the following example is L1 ball, other examples can be found at example_model_class.py
from bcg.model_init import Model

class Model_l1_ball(Model):
    def minimize(self, gradient_at_x=None):
        result = np.zeros(self.dimension)
        if gradient_at_x is None:
            result[0] = 1
        else:
            i = np.argmax(np.abs(gradient_at_x))
            result[i] = -1 if gradient_at_x[i] > 0 else 1
        return result
l1Ball = Model_l1_ball(100)  # initialize the feasible region as a L1 ball of dimension 100

# define function evaluation oracle and its gradient oracle
# the following example function is (x-shift)^2, where x is a n dimension vector

shift = np.random.randn(100)

def f(x):
    return np.dot(x - shift,x - shift)

res = BCG(f, None, l1Ball)

print('optimal solution {}'.format(res[0]))
print('dual_bound {}'.format(res[1]))
print('primal value {}'.format(res[2]))
```

# License

The code is released under the MIT license (see LICENSE).
