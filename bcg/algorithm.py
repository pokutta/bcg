import numpy as np
import time
import logging
import os
import tableprint as tp
from tabulate import tabulate
import signal

from . import utils
from . import globs


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


class Algorithm:
    def __init__(self, feasible_region, objectiveFunction, run_config):
        # parameter setting
        self.feasible_region = feasible_region
        self.objectiveFunction = objectiveFunction
        self.run_config = run_config

        self.continueRunning = True
        self.enter_sub = True  # if enter the sub algorithm
        self.func_val_all = np.array([])
        self.dual_bound_all = np.array([])
        self.wallClock_all = np.array([])
        self.start_time = None

        # for saving results
        self.info_table = []
        # there will be 10 columns of information in logging file
        self.header = ['Iteration',
                       'Type',
                       'Function Value',
                       'Primal Improve',
                       'Dual Bound',
                       '#Atoms',
                       'Iteration Time',
                       'WTime',
                       'step size',
                       'num_ls'
                       ]
        self.console_headerName, self.console_headerWidth = utils.console_header(self.header, self.run_config)
        self.it = 1  # main iteration counter
        self.x_old = self.feasible_region.solve()  # returning initial vertex via LP call
        if self.run_config['use_LPSep_oracle']:  # compute its initial bound
            c = objectiveFunction.gradient(self.x_old)
            initial_min_x = self.feasible_region.solve(c)
            phi = np.inner(c, (self.x_old - initial_min_x)) / 2
            self.dual_bound_all = np.append(self.dual_bound_all, phi)
        self.s_list = [self.x_old.copy()]  # construct initial s_list
        self.alpha = np.ones(1)  # weight vector

        self.phi_recomputedIt = None

    def update_return_info(self, val, phi):
        self.func_val_all = np.append(self.func_val_all, val)
        self.wallClock_all = np.append(self.wallClock_all, time.process_time() - self.start_time)
        self.dual_bound_all = np.append(self.dual_bound_all, phi)

    def find_index_s(self, s):
        for i in range(len(self.s_list)):
            if np.all(self.s_list[i] == s):
                return i
        return None

    def fwStep_update_alpha_S(self, step_size, index_s, s):
        if step_size == 1:
            self.s_list = [s]  # set the s list to be this atom only
            self.alpha = np.array([step_size])  # the weight for this atom is 1
        else:
            self.alpha = (1 - step_size) * self.alpha  # update weight for all the atoms originally in the s_list
            if index_s is None:  # s is new
                self.s_list.append(s)
                self.alpha = np.append(self.alpha, step_size)
            else:  # old atom
                self.alpha[index_s] += step_size

    def if_continue(self, phi, step_size, iter_type, killer):
        if phi <= self.run_config['dual_gap_acc']:
            logging.info('Exit Code 2: Achieved required dual gap accuracy, save results, and exit BCG algorithm.')
            return False
        # early stop for when reaching time limit
        if self.run_config['runningTimeLimit'] is not None:
            if self.wallClock_all[-1] > self.run_config['runningTimeLimit']:
                logging.info('Exit Code 3: Reaching time limit, save current results, and exit BCG algorithm.')
                return False
        if killer.kill_now:
            logging.error('Exit Code 0: Keyboard Interruption, save current results and exit BCG algorithm.')
            return False
        if step_size == 0 and iter_type != 'FI' and (self.it != self.phi_recomputedIt):
            logging.error('Exit Code 1: No further primal progress, save current results, and exit BCG algorithm')
            logging.error('Recomputing final dual gap.')
            return False
        return True

    def recompute_phi(self):
        est_phi = self.dual_bound_all[-1]  # estimated phi after current iteration
        c = self.objectiveFunction.gradient(self.x_old)
        s = self.feasible_region.solve(c)
        actual_phi = np.inner(-c, s - self.x_old)  # actual phi after current iteration
        phi = min(est_phi, actual_phi)
        self.dual_bound_all[-1] = phi  # rewrite phi value for after current iteration
        self.phi_recomputedIt = self.it
        return phi

    def run_algorithm(self):
        killer = GracefulKiller()
        self.start_time = time.process_time()
        with tp.TableContext(self.console_headerName, width=self.console_headerWidth) as t:
            if self.run_config['verbosity'] == 'quiet':
                logging.info('Above information is deactivated because of quiet running mode')
            while self.continueRunning:
                if self.enter_sub and self.it > 1:  # check whether to do SiGD iterations
                    current_phi = self.dual_bound_all[-1]
                    new_val = self.func_val_all[-1]

                    grad_x = self.objectiveFunction.gradient(self.x_old)
                    grad_alpha = np.dot(self.s_list, grad_x)
                    min_grad_alpha = np.min(grad_alpha)
                    max_grad_alpha = np.max(grad_alpha)

                    k = 1  # iteration counter for sub iteration
                    SIGD_improve = 'N/A'
                    while self.continueRunning and max_grad_alpha - min_grad_alpha >= current_phi/self.run_config['K']:
                        sigd_start_time = time.process_time()  # starting time of this iteration
                        old_val = new_val.copy()  # function value from last iteration

                        move_direction = -grad_alpha + np.sum(grad_alpha) / len(
                            grad_alpha)  # decomposed negative gradient
                        # compute the upper bound of step size, similar to simplex method
                        d_neg_index = np.where(move_direction < 0)[0]
                        tmp_neg = np.array([-self.alpha[j] / move_direction[j] for j in d_neg_index])
                        eta_ub = np.min(tmp_neg)
                        # line search along move direction with upper bound as eta_ub
                        step_size, num_ls, SIGD_iteration_type = \
                            utils.backtracking_ls_on_alpha(self.alpha, self.objectiveFunction,
                                                           self.s_list, eta_ub,
                                                           move_direction, self.run_config['max_lsSub'],
                                                           SIGD_improve, self.run_config['strict_dropSteps'])

                        if SIGD_iteration_type == 'PS' or k > self.run_config['max_stepsSub']:  # exit SIGD
                            self.enter_sub = False  # will remain False until a new vertex is found
                            if k > self.run_config['max_stepsSub']:
                                SIGD_iteration_type = 'PS'
                            sigd_end_time = time.process_time()  # ending time of this iteration
                            self.update_return_info(old_val, current_phi)  # didn't move
                            info = ['{}'.format(self.it), SIGD_iteration_type,
                                    '{}'.format(old_val),  # didn't move, append old val
                                    'N/A',  # here use 'N/A' instead of zero
                                    '{}'.format(current_phi),
                                    '{}'.format(len(self.s_list)),
                                    '{:.4f}'.format(sigd_end_time - sigd_start_time),
                                    '{:.4f}'.format(sigd_end_time - self.start_time),
                                    '{}'.format(step_size),
                                    '{}'.format(num_ls)
                                    ]
                            if not self.run_config['solution_only']:
                                self.info_table.append(info)
                            console_info = utils.console_info(info, self.run_config)
                            if self.run_config['verbosity'] != 'quiet':
                                t(console_info)
                            self.it += 1
                            break

                        # update alpha and check if there is vertex that needs to be dropped
                        self.alpha += step_size * move_direction
                        if step_size == eta_ub:
                            vertex_toBeDropped = np.argmin(tmp_neg)  # index in d_neg_index
                            vertex_toBeDropped_index = d_neg_index[
                                vertex_toBeDropped]  # retrieve index in the original alpha and s
                            utils.removeFromCache(self.s_list[vertex_toBeDropped_index])
                            self.s_list.pop(vertex_toBeDropped_index)
                            self.alpha = np.delete(self.alpha, vertex_toBeDropped_index)
                            # normalize
                            self.alpha *= 1 / np.sum(self.alpha)
                            SIGD_iteration_type += 'D'

                        # calculate new point and function value
                        self.x_old = np.dot(np.transpose(self.s_list), self.alpha)
                        new_val = self.objectiveFunction.evaluate(self.x_old)
                        SIGD_improve = old_val - new_val
                        grad_x = self.objectiveFunction.gradient(self.x_old)
                        grad_alpha = np.dot(self.s_list, grad_x)
                        min_grad_alpha = np.min(grad_alpha)
                        max_grad_alpha = np.max(grad_alpha)
                        sigd_end_time = time.process_time()  # ending time of this iteration
                        self.update_return_info(new_val, current_phi)
                        # save all the information
                        info = ['{}'.format(self.it), SIGD_iteration_type,
                                '{}'.format(new_val),
                                SIGD_improve,  # corresponding to function value improve
                                '{}'.format(current_phi),
                                '{}'.format(len(self.s_list)),
                                '{:.4f}'.format(sigd_end_time - sigd_start_time),
                                '{:.4f}'.format(sigd_end_time - self.start_time),
                                '{}'.format(step_size),
                                '{}'.format(num_ls)
                                ]
                        if not self.run_config['solution_only']:
                            self.info_table.append(info)
                        console_info = utils.console_info(info, self.run_config)
                        if self.run_config['verbosity'] != 'quiet':
                            t(console_info)
                        k += 1
                        self.it += 1  # so we have all iterations counter
                        self.continueRunning = self.if_continue(current_phi, step_size, SIGD_iteration_type, killer)
                if not self.continueRunning:  # break from outside loop
                    break

                fw_start_time = time.process_time()  # starting time of FW iteration
                # after sub, continue on FW or LCG step
                c = self.objectiveFunction.gradient(self.x_old)
                if self.run_config['use_LPSep_oracle']:  # call weak separation oracle
                    s, iter_type = self.feasible_region.weak_sep(c, self.x_old, True, self.dual_bound_all[-1]/float(self.run_config['K']))
                    direction = s - self.x_old
                    if iter_type != 'FIC':  # if a new vertex is found, enable enter-sub
                        self.enter_sub = True
                    if iter_type == 'FN':  # if cannot find a better point
                        direction = s - self.x_old
                        est_phi = np.inner(-c, direction)
                        if est_phi < 0:  # there is case that the LP solver stops before the optimal solution is found
                            phi = self.dual_bound_all[-1]
                        else:
                            phi = min(self.dual_bound_all[-1]/2, est_phi / 2)
                    else:  # found an improving vertex
                        phi = self.dual_bound_all[-1]
                else:  # FW
                    s = self.feasible_region.solve(c)
                    direction = s - self.x_old
                    phi = np.inner(-c, direction)
                    iter_type = 'F'

                index_s = self.find_index_s(s)
                if self.objectiveFunction.evaluate(s) <= self.objectiveFunction.evaluate(self.x_old) and \
                        len(self.s_list) >= 2*len(self.x_old):  # promote sparsity
                    step_size, num_ls = 1, 'N/A'
                else:
                    step_size, num_ls = utils.backtracking_ls_FW(self.objectiveFunction, self.x_old, c, direction,
                                                                 self.run_config['max_lsFW'])
                # update s and weight: fw update
                self.fwStep_update_alpha_S(step_size, index_s, s)
                # update x and function value
                self.x_old += step_size * direction
                new_val = self.objectiveFunction.evaluate(self.x_old)
                if self.it == 1:
                    func_val_improve = 'N/A'
                else:
                    func_val_improve = self.func_val_all[-1] - new_val
                # only for reporting
                if iter_type == 'FN':
                    reporting_phi = self.dual_bound_all[-1]  # dual bound from last iteration
                else:
                    reporting_phi = phi
                self.update_return_info(new_val, phi)  # save dual bound after current iteration
                if step_size == 0 and iter_type != 'FI' and self.phi_recomputedIt is None:
                    reporting_phi = self.recompute_phi()
                info = ['{}'.format(self.it), iter_type,
                        '{}'.format(new_val),
                        '{}'.format(func_val_improve),
                        '{}'.format(reporting_phi),
                        '{}'.format(len(self.s_list)),
                        '{:.4f}'.format(time.process_time() - fw_start_time),
                        '{:.4f}'.format(time.process_time() - self.start_time),
                        '{}'.format(step_size),
                        num_ls
                        ]
                if not self.run_config['solution_only']:
                    self.info_table.append(info)
                console_info = utils.console_info(info, self.run_config)
                if self.run_config['verbosity'] != 'quiet':
                    t(console_info)
                self.continueRunning = self.if_continue(phi, step_size, iter_type, killer)
                self.it = self.it + 1

        if not self.run_config['solution_only']:
            with open(os.path.join(globs.logging_dir, 'log.txt'), "a") as f:
                print(tabulate(self.info_table, headers=self.header, tablefmt='presto'), file=f)

        self.wallClock_all = self.wallClock_all - self.wallClock_all[0]
        if self.run_config['use_LPSep_oracle']:
            self.dual_bound_all = np.delete(self.dual_bound_all, 0)

        return self.x_old, {
            'func_val_all': self.func_val_all,
            'wallClock_all': self.wallClock_all,
            'dual_bound_all': self.dual_bound_all
        }














