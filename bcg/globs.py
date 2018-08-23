# -*- coding: utf-8 -*-
"""
Created on Fri Dec 05 21:30:01 2014

@author: spokutta
"""

######################################
# Weak Separation Oracle Parameters
######################################
nodeLimit = None
accuracyComparison = 1e-12
ggEps = 1e-08  # accuracy measure in early termination for LP solver


#################################
# Oracle Cache information
#################################
useCache = True
previousPoints = {}


###########################################
# Backtracking Line Search Parameters
###########################################
ls_tau = 0.5
ls_eps = 0.01

###############################
# Algorithm Configuration
###############################
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

#################################
# file names/directories
#################################
import os
logging_dir = os.getcwd()



#################################
# other parameters
#################################
autograd = True







