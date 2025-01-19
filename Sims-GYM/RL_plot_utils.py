from matplotlib import colors
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter1d
from plotly.subplots import make_subplots

def plot_V_single(matrix, index, title):
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

def plot_V_policy(V, Pi, title):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i / 19) for i in range(20)]
    hex_colors = [
        'rgba(' + ','.join([f'{int(np.round(rgb * 255))}' for rgb in color[:3]]) + f',{color[3]})'
        for color in colors
    ]

    colorscale = [[i / 19, color] for i, color in enumerate(hex_colors)]
    n_slices = V.shape[0]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('State Value function', 'Policy'),
        specs=[[{}, {}]],
        horizontal_spacing=0.13
    )

    for i in range(n_slices):
        visible = (i == 0)
        fig.add_trace(
            go.Heatmap(
                z=V[i, :, :],
                colorscale=colorscale,
                zmin=0, zmax=100,
                showscale=True if i == 0 else False,
                colorbar=dict(title="Values", x=0.425, len=0.8, thickness=20),
                visible=visible,
                hovertemplate='hum: %{x}<br>temp: %{y}<br>value: %{z}<extra></extra>'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(
                z=Pi[i, :, :],
                colorscale='Plasma',
                zmin=0, zmax=2,
                showscale=True if i == 0 else False,
                colorbar=dict(title="Actions", x=0.99, len=0.8, thickness=20),
                visible=visible,
                hovertemplate='hum: %{x}<br>temp: %{y}<br>action: %{z}<extra></extra>'
            ),
            row=1, col=2
        )

    steps = []
    for i in range(n_slices):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * (2 * n_slices)},
            ],
            label=str(i)
        )
        step["args"][0]["visible"][2 * i] = True
        step["args"][0]["visible"][2 * i + 1] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Growth: "},
        steps=steps
    )]

    for i in range(1, 3):
        fig.update_xaxes(title_text="Humidity", row=1, col=i, title_font={"size": 10}, title_standoff=5,
                         tickvals=np.arange(0, 11), ticktext=np.arange(0, 11))
        fig.update_yaxes(title_text="Temperature", row=1, col=i, title_font={"size": 10}, title_standoff=5,
                         tickvals=np.arange(0, 11), ticktext=np.arange(0, 11))

    fig.update_traces(showscale=True)

    sub_X = ['x1', 'x2']
    sub_Y = ['y1', 'y2']
    for j in range(2):
        for i in range(11):
            fig.add_shape(type="line", x0=i - 0.5, y0=-0.5, x1=i - 0.5, y1=10.5, line=dict(color="black", width=2),
                          xref=sub_X[j], yref=sub_Y[j])
            fig.add_shape(type="line", y0=i - 0.5, x0=-0.5, y1=i - 0.5, x1=10.5, line=dict(color="black", width=2),
                          xref=sub_X[j], yref=sub_Y[j])

    fig.update_layout(
        sliders=sliders,
        title_text=title,
        width=800, height=450,
    )
    fig.show()


def plot_Qs(Q, title):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i / 19) for i in range(20)]
    hex_colors = [
        'rgba(' + ','.join([f'{int(np.round(rgb * 255))}' for rgb in color[:3]]) + f',{color[3]})'
        for color in colors
    ]

    colorscale = [[i / 19, color] for i, color in enumerate(hex_colors)]
    n_slices = Q.shape[0]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Action 0', 'Action 1', 'Action 2'),
        specs=[[{}, {}, {}]]
    )

    columns = [
        {'col': 1, 'showscale': True, 'colorbar': None},
        {'col': 2, 'showscale': True, 'colorbar': None},
        {'col': 3, 'showscale': True, 'colorbar': {'title': "Values", 'x': 1.34, 'len': 0.8, 'thickness': 20}}
    ]

    for i in range(n_slices):
        visible = (i == 0)
        for j, settings in enumerate(columns):
            fig.add_trace(
                go.Heatmap(
                    z=Q[i, :, :, j],
                    colorscale=colorscale,
                    zmin=0, zmax=100,
                    showscale=settings['showscale'] if i == 0 else False,
                    colorbar=settings['colorbar'],
                    visible=visible,
                    hovertemplate='hum: %{x}<br>temp: %{y}<br>value: %{z}<extra></extra>'
                ),
                row=1, col=settings['col']
            )

    steps = []
    for i in range(n_slices):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * (3 * n_slices)},
            ],
            label=str(i)
        )
        step["args"][0]["visible"][3 * i] = True
        step["args"][0]["visible"][3 * i + 1] = True
        step["args"][0]["visible"][3 * i + 2] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Growth: "},
        steps=steps
    )]

    for i in range(1, 4):
        fig.update_xaxes(title_text="Humidity", row=1, col=i, title_font={"size": 10}, title_standoff=5,
                         tickvals=np.arange(0, 11), ticktext=np.arange(0, 11))
        fig.update_yaxes(title_text="Temperature", row=1, col=i, title_font={"size": 10}, title_standoff=5,
                         tickvals=np.arange(0, 11), ticktext=np.arange(0, 11))

    fig.update_traces(showscale=True)

    sub_X = ['x1', 'x2', 'x3']
    sub_Y = ['y1', 'y2', 'y3']
    for j in range(3):
        for i in range(11):
            fig.add_shape(type="line", x0=i - 0.5, y0=-0.5, x1=i - 0.5, y1=10.5, line=dict(color="black", width=2),
                          xref=sub_X[j], yref=sub_Y[j])
            fig.add_shape(type="line", y0=i - 0.5, x0=-0.5, y1=i - 0.5, x1=10.5, line=dict(color="black", width=2),
                          xref=sub_X[j], yref=sub_Y[j])

    # Configura il layout
    fig.update_layout(
        sliders=sliders,
        title_text=title,
        width=1350, height=450,
    )

    fig.show()


def plot_data_conditions(V, Pi,conditions, title):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i / 19) for i in range(20)]
    hex_colors = [
        'rgba(' + ','.join([f'{int(np.round(rgb * 255))}' for rgb in color[:3]]) + f',{color[3]})'
        for color in colors
    ]

    colorscale = [[i / 19, color] for i, color in enumerate(hex_colors)]
    n_slices = V.shape[0]

    custom_colorscale = [[-2, 'red'],[-1, 'orange'],[0, 'yellow'],[1, 'green']]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('State value function', 'Policy', 'Relative growth'),
        specs=[[{}, {}, {}]],
        horizontal_spacing = 0.09
    )

    columns = [
        {'data':V,         'col': 1, 'colorscale':colorscale,'showscale': True,'zmin':0,'zmax':100,'colorbar': {'title': "Values", 'x': 0.265, 'len': 0.8, 'thickness': 18,'tickfont': {'size': 10}}},
        {'data':Pi,        'col': 2, 'colorscale':'plasma',  'showscale': True,'zmin':0,'zmax':2, 'colorbar': {'title': "Actions", 'x': 0.63, 'len': 0.8, 'thickness': 18,'tickfont': {'size': 10}}},
        {'data':conditions,'col': 3, 'colorscale':'RdBu',  'showscale': True,'zmin':-2,'zmax':1, 'colorbar': {'title': "Growth", 'x': 0.995, 'len': 0.8, 'thickness': 18,'tickfont': {'size': 10}}}
    ]

    for i in range(n_slices):
        visible = (i == 0)
        for j, settings in enumerate(columns):
            fig.add_trace(
                go.Heatmap(
                    z=columns[j]['data'][i, :, :],
                    colorscale=columns[j]['colorscale'],
                    zmin=columns[j]['zmin'], zmax=columns[j]['zmax'],
                    showscale=settings['showscale'] if i == 0 else False,
                    colorbar=settings['colorbar'],
                    visible=visible,
                    hovertemplate='hum: %{x}<br>temp: %{y}<br>value: %{z}<extra></extra>'
                ),
                row=1, col=settings['col']
            )

    steps = []
    for i in range(n_slices):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * (3 * n_slices)},
            ],
            label=str(i)
        )
        step["args"][0]["visible"][3 * i] = True
        step["args"][0]["visible"][3 * i + 1] = True
        step["args"][0]["visible"][3 * i + 2] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Growth: "},
        steps=steps
    )]

    for i in range(1, 4):
        fig.update_xaxes(title_text="Humidity", row=1, col=i, title_font={"size": 10}, title_standoff=5,
                         tickvals=np.arange(0, 11), ticktext=np.arange(0, 11))
        fig.update_yaxes(title_text="Temperature", row=1, col=i, title_font={"size": 10}, title_standoff=5,
                         tickvals=np.arange(0, 11), ticktext=np.arange(0, 11))

    fig.update_traces(showscale=True)

    sub_X = ['x1', 'x2', 'x3']
    sub_Y = ['y1', 'y2', 'y3']
    for j in range(3):
        for i in range(11):
            fig.add_shape(type="line", x0=i - 0.5, y0=-0.5, x1=i - 0.5, y1=10.5, line=dict(color="black", width=2),
                          xref=sub_X[j], yref=sub_Y[j])
            fig.add_shape(type="line", y0=i - 0.5, x0=-0.5, y1=i - 0.5, x1=10.5, line=dict(color="black", width=2),
                          xref=sub_X[j], yref=sub_Y[j])

    # Configura il layout
    fig.update_layout(
        sliders=sliders,
        title_text=title,
        width=1050, height=450,
    )

    fig.show()