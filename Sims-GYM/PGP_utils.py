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