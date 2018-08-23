# conditional autograd import
import importlib.util
spec = importlib.util.find_spec("autograd")
if spec is None:
    import numpy as np
else:
    import autograd.numpy as np
    from autograd import grad



import logging

from .objective_functions.obj_function import ObjectiveFunction
from .algorithm import Algorithm
from . import utils
from . import globs


#TODO: Logging shouldn't be configured when run as a module, it is the
#task of the main application.
logging.basicConfig(level=logging.INFO,
                    format='%(message)s')


def init_model(model):
    if isinstance(model, str):  # LP file
        from .LPsolver import initModel_fromFile
        feasible_region = initModel_fromFile(model)
    else:  # user-defined model class
        feasible_region = model
    return feasible_region


def BCG(f, f_grad, model, run_config=None):
    # reset global statistic variables
    utils.reset_cache()

    feasible_region = init_model(model)
    dim = feasible_region.dimension

    # autograd objective function is possible
    if f_grad is None:
        spec = importlib.util.find_spec("autograd")
        if spec is None or globs.autograd is False:
            objectiveFunction = ObjectiveFunction(f, f_grad)
        else:
            logging.info("Trying to use autograd to compute gradient...")
            try:
                objectiveFunction = ObjectiveFunction(f, grad(f))
            except:
                logging.info("Autograd failed. Using numerical approximation.")
                objectiveFunction = ObjectiveFunction(f, f_grad)
                pass
    else:
        objectiveFunction = ObjectiveFunction(f, f_grad)

    logging.info('Dimension of feasible region: {}'.format(dim))

    if f_grad is not None:  # check the user provided gradient oracle
        logging.info('Checking validity of function gradient...')
        sample_checkpoint = np.random.rand(dim)*10.0
        check_res = objectiveFunction.check_gradient(sample_checkpoint)
        assert check_res, 'gradient check error is {}'.format(check_res)

    if run_config is None:  # make it global
        run_config = globs.run_config
    # run algorithm
    algorithmRun = Algorithm(feasible_region, objectiveFunction, run_config)
    optimal_x, result_dict = algorithmRun.run_algorithm()
    dual_bound = result_dict['dual_bound_all'][-1]
    primal_val = result_dict['func_val_all'][-1]

    if not globs.run_config['solution_only']:  # save the model pickle file in current working folder
        import pickle
        import os
        pickle.dump(result_dict, open(os.getcwd() + '/model_result.p', 'wb'))
    return optimal_x, dual_bound, primal_val



