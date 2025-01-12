import numpy as np
import pygame

def pars_loader(N_episodes,alpha,gamma,eps,pd,growth_range,factor_range,ci_range,action_set=[0,1,2]):
    pars= {
    'N_episodes':N_episodes,
    'alpha': alpha,  # Learning rate
    'gamma': gamma,  # Discount factor
    'eps': eps,    # Epsilon greedy
    'pd':pd,     # Probability to not decay
    'growth_range': growth_range,
    'factor_range': factor_range,
    'ci_range': np.arange(ci_range[0], ci_range[1]),
    'action_set': action_set,
    'opt_range': {
        1: {'Tem': [[3, 4, 5], [2, 6, 7, 8], [1, 9], [0, 10]], 'Hum': [[6, 7, 8], [2, 3, 4, 5], [1, 9], [0, 10]]},
        2: {'Tem': [[6, 7, 8], [4, 5], [1, 2, 3, 9], [0, 10]], 'Hum': [[4, 5, 6], [2, 3, 7, 8], [1, 9], [0, 10]]},
        3: {'Tem': [[5, 6, 7], [3, 4, 8], [1, 2, 9], [0, 10]], 'Hum': [[6, 7, 8], [3, 4, 5], [1, 2, 9], [0, 10]]},
        4: {'Tem': [[6, 7, 8], [4, 5, 9], [1, 2, 3], [0, 10]], 'Hum': [[7, 8, 9], [4, 5, 6], [1, 2, 3], [0, 10]]}}
    }
    pars['nstates']=(pars['growth_range']+1,pars['factor_range']+1,pars['factor_range']+1)
    return pars

