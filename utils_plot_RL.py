from matplotlib import colors
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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
        zmin=-100,
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