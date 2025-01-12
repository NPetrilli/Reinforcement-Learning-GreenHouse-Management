from matplotlib import colors
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter1d

def plot_state_value_function_single(matrix, index, title):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 2, 1)
    matrix_flipped = np.flipud(matrix[index, :, :])
    pc = ax.imshow(matrix_flipped, cmap='tab20b', vmin=-100, vmax=100, aspect='equal')
    cbar = fig.colorbar(pc, pad=0.2)
    cbar.set_ticks(np.arange(-100, 101, 10))
    ax.set_title(title + f'(Growth={index})', fontsize=13)
    ax.set(xlabel='Temperature', ylabel='Humidity')
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(0, 11, 1))

    ax.set_xticklabels(np.arange(0, 11, 1))
    ax.set_yticklabels(np.arange(10, -1, -1))

    ax.set_xticks(np.arange(-.5, matrix_flipped.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, matrix_flipped.shape[0], 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    plt.show()

def plot_value_dyn(matrix, start_index, end_index, title=None):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i / 19) for i in range(20)]
    hex_colors = [
        'rgba(' + ','.join([f'{int(np.round(rgb * 255))}' for rgb in color[:3]]) + f',{color[3]})'
        for color in colors
    ]

    colorscale = [[i / 19, color] for i, color in enumerate(hex_colors)]

    fig = px.imshow(
        np.flipud(matrix[start_index, :, :]),
        labels=dict(x="Temperature", y="Humidity", color="Value"),
        x=np.arange(11),
        y=np.arange(11),
        color_continuous_scale=colorscale,
        zmin=0,
        zmax=100,
        title=title
    )

    for i in range(11):
        fig.add_shape(type="line", x0=i - 0.5, y0=-0.5, x1=i - 0.5, y1=10.5, line=dict(color="black", width=2))
        fig.add_shape(type="line", y0=i - 0.5, x0=-0.5, y1=i - 0.5, x1=10.5, line=dict(color="black", width=2))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(11),
            ticktext=[str(x) for x in range(11)],
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(11),
            ticktext=[str(x) for x in range(10, -1, -1)]
        ),
        title=dict(text=title, x=0.5, y=0.85),
        width=500,
        height=500
    )
    fig.update_coloraxes(colorbar=dict(x=1))

    fig.update_layout(
        sliders=[{
            "pad": {"t": 15},
            "currentvalue": {"prefix": "Growth: "},
            "steps": [
                {
                    "method": "restyle",
                    "args": [{"z": [np.flipud(matrix[i, :, :])]}],
                    "label": str(i)
                } for i in range(start_index, end_index + 1)
            ]
        }]
    )

    return fig

def plot_policy_dyn(V, start_index, end_index):
    color_mapping={0: 'blue', 1: 'yellow', 2: 'red'}
    colorscale = [[k / (len(color_mapping) - 1), v] for k, v in enumerate(color_mapping.values())]

    fig = px.imshow(
        np.flipud(V[start_index, :, :]),
        labels=dict(x="Temperature", y="Humidity", color="Value"),
        x=np.arange(V.shape[2]),
        y=np.arange(V.shape[1]),
        color_continuous_scale=colorscale,
        zmin=min(color_mapping.keys()),
        zmax=max(color_mapping.keys()),
        title="Policy"
    )

    for i in range(V.shape[2]):
        fig.add_shape(type="line", x0=i-0.5, y0=-0.5, x1=i-0.5, y1=V.shape[1]-0.5, line=dict(color="black", width=2))
        fig.add_shape(type="line", y0=i-0.5, x0=-0.5, y1=i-0.5, x1=V.shape[2]-0.5, line=dict(color="black", width=2))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(V.shape[2]),
            ticktext=[str(x) for x in range(V.shape[2])],
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(V.shape[1]),
            ticktext=[str(x) for x in range(V.shape[1]-1, -1, -1)]
        ),
        title=dict(text="Policy", x=0.5, y=0.85),
        width=500,
        height=500
    )
    fig.update_coloraxes(colorbar=dict(x=1))

    fig.update_layout(
        sliders=[{
            "pad": {"t": 15},
            "currentvalue": {"prefix": "Index: "},
            "steps": [
                {
                    "method": "restyle",
                    "args": [{"z": [np.flipud(V[i, :, :])]}],
                    "label": str(i)
                } for i in range(start_index, end_index + 1)
            ]
        }]
    )

    return fig


def training_plot(data_collection, win_length):
    learning_result = []
    learning_length = []
    for method in data_collection:
        if len(method[0]) >= win_length:
            rewards = (np.convolve(method[0], np.ones(win_length), mode="valid") / win_length)
            length = (np.convolve(method[1], np.ones(win_length), mode="same") / win_length)
            learning_result.append([rewards, method[2]])
            learning_length.append([length, method[2]])

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Episode rewards
    for result in learning_result:
        ax1.plot(range(len(result[0])), result[0], label=result[1])
    ax1.set_title("Episode rewards")
    ax1.legend()
    ax1.grid(True)
    # Episode lengths
    for length in learning_length:
        ax2.plot(range(len(length[0])), length[0], label=length[1])
    ax2.set_title("Episode lengths")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()




# def training_plot2(data_collection, win_length, pars, sample_rate=1):
#     fig, ax = plt.subplots(figsize=(6, 4))  # Usando un solo asse
#     # Processa i dati per ciascun metodo
#     for episode_rewards, _, label in data_collection:  # Ignora episode_lengths
#         if pars['N_episodes'] is not None and pars['N_episodes'] < len(episode_rewards):
#             episode_rewards = episode_rewards[:pars['N_episodes']]
#
#         if len(episode_rewards) >= win_length:
#             rewards = uniform_filter1d(episode_rewards, size=win_length, mode='reflect') / win_length
#
#             sampled_indices = np.arange(0, len(rewards), sample_rate)
#             sampled_rewards = rewards[sampled_indices]
#
#             # Plot dei dati campionati
#             ax.plot(sampled_indices, sampled_rewards, label=f"{label}")
#
#     ax.set_title("Episode Returns")
#     ax.legend()
#     ax.grid(True)
#
#     plt.tight_layout()
#     plt.show()


def training_plot2(data_collection, win_length, sample_rate=1):
    fig, ax = plt.subplots(figsize=(6, 4))  # Usando un solo asse
    # Processa i dati per ciascun metodo
    for episode_rewards, _, label in data_collection:  # Ignora episode_lengths
        if len(episode_rewards) >= win_length:
            # Calcolo efficiente della media mobile
            rewards = uniform_filter1d(episode_rewards, size=win_length, mode='reflect') / win_length

            # Campionamento dei dati per ridurre la densità dei plot
            sampled_indices = np.arange(0, len(rewards), sample_rate)
            sampled_rewards = rewards[sampled_indices]

            # Plot dei dati campionati
            ax.plot(sampled_indices, sampled_rewards, label=f"{label}")

    # Imposta titoli ed etichette
    ax.set_title("Episode Returns")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()







# def training_plot2(data_collection, win_length, sample_rate=1):
#     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
#     # Process each method's data
#     for episode_rewards, episode_lengths, label in data_collection:
#         if len(episode_rewards) >= win_length:
#             # Efficiently compute moving average
#             rewards = uniform_filter1d(episode_rewards, size=win_length, mode='reflect') / win_length
#             lengths = uniform_filter1d(episode_lengths, size=win_length, mode='reflect') / win_length
#
#             # Sample data to reduce plot density
#             sampled_indices = np.arange(0, len(rewards), sample_rate)
#             sampled_rewards = rewards[sampled_indices]
#             sampled_lengths = lengths[sampled_indices]
#
#             # Plotting the sampled data
#             ax1.plot(sampled_indices, sampled_rewards, label=f"{label}")
#             ax2.plot(sampled_indices, sampled_lengths, label=f"{label}")
#
#     # Set titles and labels
#     ax1.set_title("Episode Returns")
#     ax1.legend()
#     ax1.grid(True)
#
#     ax2.set_title("Episode Lengths")
#     ax2.legend()
#     ax2.grid(True)
#
#     plt.tight_layout()
#     plt.show()



