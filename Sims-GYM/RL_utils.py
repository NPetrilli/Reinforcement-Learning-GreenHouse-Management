import numpy as np
import pygame

def pars_loader(N_episodes,alpha,gamma,eps,pt,ph,growth_range,factor_range,ci_range,ci_g_range,ci_range_sim,action_set=[0,1,2]):
    pars= {
    'N_episodes':N_episodes,
    'alpha': alpha,  # Learning rate
    'gamma': gamma,  # Discount factor
    'eps': eps,    # Epsilon greedy
    'pt':pt,     # Probability to not decay
    'ph':ph,
    'ci_range_sim':ci_range_sim,
    'growth_range': growth_range,
    'factor_range': factor_range,
    'ci_range': np.arange(ci_range[0], ci_range[1]),
    'ci_g_range':ci_g_range,
    'action_set': action_set,
    'opt_range': {
        1: {'Tem': [[3, 4, 5], [2, 6, 7, 8], [1, 9], [0, 10]], 'Hum': [[6, 7, 8], [2, 3, 4, 5], [1, 9], [0, 10]]},
        2: {'Tem': [[6, 7, 8], [4, 5], [1, 2, 3, 9], [0, 10]], 'Hum': [[4, 5, 6], [2, 3, 7, 8], [1, 9], [0, 10]]},
        3: {'Tem': [[5, 6, 7], [3, 4, 8], [1, 2, 9], [0, 10]], 'Hum': [[6, 7, 8], [3, 4, 5], [1, 2, 9], [0, 10]]},
        4: {'Tem': [[6, 7, 8], [4, 5, 9], [1, 2, 3], [0, 10]], 'Hum': [[7, 8, 9], [4, 5, 6], [1, 2, 3], [0, 10]]}},
    'file_name':'Results/Data_N_'+str(N_episodes)+'_alpha_'+str(alpha)+'_gamma_'+str(gamma)+'_eps_'+str(eps)+'_pt_'+str(pt)+'_ph_'+str(ph)+'.pickle'
    }
    pars['nstates']=(pars['growth_range']+1,pars['factor_range']+1,pars['factor_range']+1)

    return pars

def load_images(base_path, count=None, scale=None):
    if count is None:
        try:
            img = pygame.image.load(f"{base_path}.png")
            if scale is not None:
                img_size = (int(img.get_width()/scale[0]), int(img.get_height()/scale[1]))
                img = pygame.transform.scale(img, img_size)
            return img
        except pygame.error as e:
            print(f"Error loading image from {base_path}.png: {e}")
            return None
    else:
        images = []
        for i in range(count):
            try:
                img = pygame.image.load(f'{base_path}{i}.png')
                if scale is not None:
                    img_size = (int(img.get_width()/scale[0]), int(img.get_height() / scale[1]))
                    img = pygame.transform.scale(img, img_size)
                images.append(img)
            except pygame.error as e:
                print(f"Error loading image {base_path}{i}.png: {e}")
        return images

def render_text(window, font, text, position, color=(255, 255, 255)):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=position)
    window.blit(text_surface, text_rect)

def get_color_for_value(value, range_lists):
    colors = [(39, 174, 96), (241, 196, 15), (230, 126, 34), (231, 76, 60)]
    for idx, range_list in enumerate(range_lists):
        if value in range_list:
            return colors[idx]
    return (0, 0, 0)  # default color


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

def recover_policy_V(Q,pars):
    policy=np.zeros(pars['nstates'],dtype=int)
    V=np.zeros(pars['nstates'])
    for i in range(pars['nstates'][0]):
        for j in range(pars['nstates'][1]):
            for k in range(pars['nstates'][2]):
             policy[i,j,k]=np.argmax(Q[i,j,k,:])
             V[i,j,k]=np.max(Q[i,j,k,:])
    return policy,V

def epsilon_greedy(pars,state,Q):
    if np.random.random() < pars['eps']:
     a = np.random.choice(pars['action_set'])
    else:
     a = np.argmax(Q[state[0],state[1],state[2],:])
    return a

def growth_to_matrix(growth, temp, hum, range_tem, range_hum):
    #At least one red: bad condition to growth -2
    if temp in range_tem[3] or hum in range_hum[3]:
      growth+=-2
    #At least one orange : poor condition to growth -1
    elif temp in range_tem[2] or hum in range_hum[2]:
      growth+=-1
    #At least one yellow : good condition to growth +0
    elif temp in range_tem[1] or hum in range_hum[1]:
     growth+=0
    else:
     #Only green : Optimal condition to growth +1
     growth+=1
    return growth


def condition_to_matrix(pars):
    gr = np.zeros((pars['nstates'][0], pars['nstates'][1], pars['nstates'][2]))
    for i in np.arange(1, 5):
        for j in range(pars['nstates'][1]):
            for k in range(pars['nstates'][2]):
                range_tem, range_hum = pars['opt_range'][i]['Tem'], pars['opt_range'][i]['Hum']
                gr[i, j, k] = growth_to_matrix(0, j, k, range_tem, range_hum)
    return gr
def make_episode_env(env, policy, pars,sim=False, x0=None):
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
        if np.random.random() < pars['eps'] and sim==False:
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


    # 'opt_range': {
    #     1: {'Tem': [[6], [5, 7], [1,2,3,4,8,9], [0, 10]], 'Hum': [[4], [3, 5], [1,2,6,7,8,9], [0, 10]]},
    #     2: {'Tem': [[4], [3, 5], [1, 2, 6,7,8,9], [0, 10]], 'Hum': [[7], [6,8], [1,2,3,4,5,9], [0, 10]]},
    #     3: {'Tem': [[8], [7], [1, 2,3,4,5,6, 9], [0, 10]], 'Hum': [[3], [2, 4], [1,5,6,7,8, 9], [0, 10]]},
    #     4: {'Tem': [[3], [2,4], [1,5,6,7,8,9], [0, 10]], 'Hum': [[8], [7], [1, 2, 3,4,5,6,9], [0, 10]]}}
    # }