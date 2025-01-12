import numpy as np

def make_episode(env, policy, pars, x0=None):
    """
    :param env: Environment to render
    :param policy: Policy to use
    :param pars: Parameters to use
    :param x0: Initial condition to use
    :return: sequence of state, sequence of action, sequence of reward
    """
    seq_state = []
    seq_action = []
    seq_reward = []

    # Stato iniziale
    state, info = env.reset(options=x0)
    seq_state.append(state)
    done = False
    env.render()

    while not done:
        if np.random.random() < pars['eps']:
            a = np.random.choice(pars['action_set'])
        else:
            a = policy[seq_state[-1]]

        state, reward, done, _,_ = env.step(a)
        env.render()
        seq_state.append(state)
        seq_action.append(a)
        seq_reward.append(reward)

    return seq_state, seq_action, seq_reward


def state_value_MC(st_list, ac_list, re_list, pars):
    '''
    :param st_list: List of state's episode
    :param ac_list: List of action's episode
    :param re_list: List of reward's episode
    :param pars: Parameter to Use
    :return: State value function for that policy
    '''
    V = np.zeros(pars['nstates'])

    counter = np.zeros(pars['nstates'])

    for st, at, rt in zip(st_list, ac_list, re_list):
        G = 0.0

        T = len(st) - 1

        for t in range(T - 1, -1, -1):
            s = st[t]
            r = rt[t]

            G = pars['gamma'] * G + r

            if s not in st[:t]:
                counter[s] += 1
                V[s] += 1 / counter[s] * (G - V[s])

    return V


def action_state_value_MC(st_list, ac_list, re_list, pars, Q0=None, counter_s_a_0=None):
    '''
    :param st_list: List of state's episode
    :param ac_list: List of action's episode
    :param re_list: List of reward's episode
    :param pars:  Parameter to Use
    :return: State action value function for that policy
    '''
    Q = Q0.copy()
    counter_s_a = counter_s_a_0.copy()

    for st, at, rt in zip(st_list, ac_list, re_list):
        G = 0.0
        pair_s_a = list(zip(st[:-1], at))

        T = len(st) - 1

        for t in range(T - 1, -1, -1):
            s = st[t]
            a = at[t]
            r = rt[t]

            G = pars['gamma'] * G + r

            if (s, a) not in pair_s_a[:t]:
                counter_s_a[s[0], s[1], s[2], a] += 1
                Q[s[0], s[1], s[2], a] += 1 / counter_s_a[s[0], s[1], s[2], a] * (G - Q[s[0], s[1], s[2], a])

    return Q, counter_s_a


#
# for _ in tqdm(range(pars['N_episodes'])):
#     seq_state,seq_action,seq_reward=make_episode(env,policy,pars)
# env.close()

# x0={'g': 2, 'temp': 5, 'hum': 3}



# opt_range = {}
# #Germinazione
# opt_range[1] = {'Tem': [[4,5,6], [2, 3, 7], [1,8,9], [0, 10]], 'Hum': [[6, 7, 8], [4, 5,9], [1,2,3], [0,10]]}
# #Crescita iniziale
# opt_range[2] = {'Tem': [[6, 7, 8], [4, 5,9], [1,2,3], [0,10]], 'Hum': [[4,5,6], [2,3, 7], [1,8,9], [0, 10]]}
# #Crescita avanzanta
# opt_range[3] = {'Tem': [[6, 7,8], [4, 5, 9], [1, 2, 3], [0,10]], 'Hum': [[5,6, 7], [3, 4, 8], [1, 2, 9], [0, 10]]}
# #Maturazione
# opt_range[4] = {'Tem': [[4, 5, 6], [2, 3, 7, 8], [1, 9], [0, 10]], 'Hum': [[5,6,7], [3,4,8], [1, 2, 9], [0, 10]]}
#
# pars['opt_range'] = opt_range