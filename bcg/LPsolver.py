"""LP model based on Gurobi solver."""

import os
import logging
import numpy as np
from gurobipy import GRB, read, Column


from .model_init import Model
from . import globs


class ModelGurobi(Model):
    """LP model implemented via Gurobi."""

    def __init__(self, model):
        super().__init__(len(model.getVars()))
        self.model = model

    def Gurobi_solve(self, cc, callback, goal):
        """Find good solution for cc with optimality callback."""
        m = self.model
        for it, v in enumerate(m.getVars()):
            v.setAttr(GRB.attr.Obj, cc[it])

        m.update()
        m.optimize(lambda mod, where: callback(mod, where, goal))
        return np.array([v.x for v in m.getVars()], dtype=float)[:]

    def minimize(self, cc=None):
        if cc is None:  # to find initial solution
            return self.minimize(np.zeros(self.dimension))
        else:
            return self.Gurobi_solve(cc, fakeCallback, GRB.INFINITY)

    def augment(self, cc=None, x=None, goal=None):
        # TODO: examine this later!!!
        self.model.params.cutoff = GRB.INFINITY

        if cc is None or x is None:
            cc = np.zeros(self.dimension)

        return self.Gurobi_solve(cc, thresholdCallback,
                                 goal if goal is not None
                                 else GRB.INFINITY)


def thresholdCallback(model, where, value):
    if where == GRB.Callback.MIPSOL:
        # x = model.cbGetSolution(model.getVars())
        # logging.info 'type of x: ' + str(type(x))
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if obj < value - globs.ggEps:
            logging.debug('early termination with objective value :{}'.format(obj))
            logging.debug('which is better than {}'.format(value - globs.ggEps))
            model.terminate()

    if where == GRB.Callback.MIP:
        objBnd = model.cbGet(GRB.Callback.MIP_OBJBND)

        if objBnd >= value + globs.ggEps:
            model.terminate()
            pass


def fakeCallback(model, where, value):

    if where == GRB.Callback.MIPSOL:
        # x = model.cbGetSolution(model.getVars())
        # logging.info 'type of x: ' + str(type(x))
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if obj < value - globs.ggEps:
            logging.debug('early termination with objective value :{}'.format(obj))
            logging.debug('which is better than {}'.format(value - globs.ggEps))
            # model.terminate()

    if where == GRB.Callback.MIP:
        objBnd = model.cbGet(GRB.Callback.MIP_OBJBND)

        if objBnd >= value + globs.ggEps:
            # model.terminate()
            pass


def initModel_fromFile(modelFilename, addCubeConstraints=False, transform_to_equality=False):
    logging.info('Initializing the model .....')
    model = read(modelFilename)
    model.setParam('OutputFlag', False)
    model.params.TimeLimit = globs.run_config['LPsolver_timelimit']
    model.params.threads = 4
    model.params.MIPFocus = 0

    if globs.nodeLimit is not None:
        model.params.nodelimit = globs.nodeLimit

    # -1 for maximization
    # model.modelSense = -1

    model.update()

    logging.info('modelSense: ' + str(model.modelSense))
    logging.info('outputFlag: ' + str(model.params.OutputFlag))

    if addCubeConstraints:
        counter = 0
        for v in model.getVars():
            model.addConstr(v <= 1, 'unitCubeConst' + str(counter))
            counter += 1

    model.update()

    if transform_to_equality:
        for c in model.getConstrs():
            sense = c.sense
            if sense == GRB.GREATER_EQUAL:
                model.addVar(obj=0, name="ArtN_" + c.constrName, column=Column([-1], [c]))
            if sense == GRB.LESS_EQUAL:
                model.addVar(obj=0, name="ArtP_" + c.constrName, column=Column([1], [c]))
            c.sense = GRB.EQUAL

    model.update()
    return ModelGurobi(model)

