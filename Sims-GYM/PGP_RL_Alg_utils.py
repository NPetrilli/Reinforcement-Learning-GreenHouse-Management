import numpy as np


def init_policy(pars):
    """
    :param pars: Algorithm parameters
    :return: Return a random parameters
    """
    policy=np.zeros(pars['nstates'],dtype=int)
    for i in range(pars['nstates'][0]):
        for j in range(pars['nstates'][1]):
            for k in range(pars['nstates'][2]):
             policy[i,j,k]=np.random.choice(pars['action_set'])
    return policy


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
    state, info = env.reset()
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