

# Selezioniamo data1, data2, data3 usando l'indice fornito
data1 = Q[1, :, :, 0]
data2 = Q[1, :, :, 1]
data3 = Q[1, :, :, 2]

# Creazione del plot 3D per data1, data2, data3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Coordinate x, y per data1, data2, data3
x = np.arange(data1.shape[0])
y = np.arange(data1.shape[1])
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()
z = np.zeros(data1.size)

# Altezza delle colonne
dx = 0.25  # Larghezza ridotta per permettere la rappresentazione affiancata
dy = 0.25
dz1 = data1.flatten()
dz2 = data2.flatten()
dz3 = data3.flatten()

# Posizione aggiustata su asse x per evitare sovrapposizioni
ax.bar3d(x, y, z, dx, dy, dz1, color='b')
ax.bar3d(x + 0.3, y, z, dx, dy, dz2, color='r')
ax.bar3d(x + 0.6, y, z, dx, dy, dz3, color='g')

# Etichette e titolo
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Valore')
plt.title('Plot 3D di data1, data2, data3 affiancati')

plt.show()



#####################################################

data1 = Q[1, :, :, 0]
data2 = Q[1, :, :, 1]
data3 = Q[1, :, :, 2]

# Creazione del plot 3D per data1, data2, data3
fig = plt.figure(figsize=(18, 6))  # Larghezza aumentata per accomodare tre plot

# Coordinate x, y per i plot
x = np.arange(data1.shape[0])
y = np.arange(data1.shape[1])
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()
z = np.zeros(data1.size)

# Altezza delle colonne e larghezza ridotta
dx = 0.75  # Più larghe per riempire ciascun plot singolarmente
dy = 0.75
dz1 = data1.flatten()
dz2 = data2.flatten()
dz3 = data3.flatten()

# Plot per data1
ax1 = fig.add_subplot(131, projection='3d')
ax1.bar3d(x, y, z, dx, dy, dz1, color='b')
ax1.set_title('Data1')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Valore')

# Plot per data2
ax2 = fig.add_subplot(132, projection='3d')
ax2.bar3d(x, y, z, dx, dy, dz2, color='r')
ax2.set_title('Data2')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Valore')

# Plot per data3
ax3 = fig.add_subplot(133, projection='3d')
ax3.bar3d(x, y, z, dx, dy, dz3, color='g')
ax3.set_title('Data3')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Valore')

plt.tight_layout()
plt.show()
##########################################################à

from matplotlib.colors import Normalize

# Assumiamo che Q sia definito come un array 4D da qualche parte nel tuo codice

# Selezioniamo data1, data2, data3 usando l'indice fornito
data1 = Q[1, :, :, 0]
data2 = Q[1, :, :, 1]
data3 = Q[1, :, :, 2]

# Creazione del plot 3D per data1, data2, data3
fig = plt.figure(figsize=(18, 6))

# Coordinate x, y per i plot
x = np.arange(data1.shape[0])
y = np.arange(data1.shape[1])
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()
z = np.zeros(data1.size)

# Altezza delle colonne e larghezza ridotta
dx = 0.75  # Più larghe per riempire ciascun plot singolarmente
dy = 0.75
dz1 = data1.flatten()
dz2 = data2.flatten()
dz3 = data3.flatten()

# Normalizzatore per i colori
norm = Normalize(vmin=min(np.min(dz1), np.min(dz2), np.min(dz3)), vmax=max(np.max(dz1), np.max(dz2), np.max(dz3)))
cmap = plt.get_cmap('tab20b')

# Plot per data1
ax1 = fig.add_subplot(131, projection='3d')
colors = cmap(norm(dz1))
ax1.bar3d(x, y, z, dx, dy, dz1, color=colors)
ax1.set_title('Data1')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Valore')

# Plot per data2
ax2 = fig.add_subplot(132, projection='3d')
colors = cmap(norm(dz2))
ax2.bar3d(x, y, z, dx, dy, dz2, color=colors)
ax2.set_title('Data2')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Valore')

# Plot per data3
ax3 = fig.add_subplot(133, projection='3d')
colors = cmap(norm(dz3))
ax3.bar3d(x, y, z, dx, dy, dz3, color=colors)
ax3.set_title('Data3')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Valore')

plt.tight_layout()
plt.show()



# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np
#
# def plot_V_policy(V=None, policy=None, index_min=0, index_max=None):
#     subplot_titles = []
#     if V is not None:
#         subplot_titles.append("Value")
#         cmap = plt.get_cmap('tab20b')
#         colors = [cmap(i / 19) for i in range(20)]
#         hex_colors = [
#             'rgba(' + ','.join([f'{int(np.round(rgb * 255))}' for rgb in color[:3]]) + f',{color[3]})'
#             for color in colors
#         ]
#
#         colorscale_V = [[i / 19, color] for i, color in enumerate(hex_colors)]
#     if policy is not None:
#         subplot_titles.append("Policy")
#
#     fig = make_subplots(rows=1, cols=len(subplot_titles), subplot_titles=subplot_titles)
#
#
#     if V is not None:
#
#         fig.add_trace(
#             go.Heatmap(
#                 z=np.flipud(V[index_min, :, :]),
#                 x=np.arange(V.shape[2]),  # Assuming square matrix
#                 y=np.arange(V.shape[1]),
#                 colorscale=colorscale_V,
#                 colorbar=dict(len=1, x=0.45),
#                 zmin=0,
#                 zmax=100,
#                 showscale=True
#             ),
#             row=1, col=1
#         )
#
#     if policy is not None:
#         fig.add_trace(
#             go.Heatmap(
#                 z=np.flipud(policy[index_min, :, :]),
#                 colorscale='Cividis',
#                 zmin=policy.min(),
#                 zmax=policy.max(),
#                 colorbar=dict(x=1)
#             ),
#             row=1, col=len(subplot_titles)
#         )
#
#     # Aggiungere linee per formare una griglia definita
#     if V is not None:
#         for i in range(V.shape[2] + 1):
#             fig.add_shape(type="line", x0=i-0.5, y0=-0.5, x1=i-0.5, y1=V.shape[1]-0.5, line=dict(color="black", width=1), row=1, col=1)
#             fig.add_shape(type="line", y0=i-0.5, x0=-0.5, y1=i-0.5, x1=V.shape[2]-0.5, line=dict(color="black", width=1), row=1, col=1)
#
#     if policy is not None:
#         for i in range(policy.shape[2] + 1):
#             fig.add_shape(type="line", x0=i-0.5, y0=-0.5, x1=i-0.5, y1=policy.shape[1]-0.5, line=dict(color="black", width=1), row=1, col=len(subplot_titles))
#             fig.add_shape(type="line", y0=i-0.5, x0=-0.5, y1=i-0.5, x1=policy.shape[2]-0.5, line=dict(color="black", width=1), row=1, col=len(subplot_titles))
#
#     fig.update_layout(
#         height=500,
#         width=1000,
#         sliders=[{
#             "pad": {"t": 50},
#             "currentvalue": {"prefix": "Index: "},
#             "steps": [
#                 {
#                     "method": "update",
#                     "args": [{"z": [np.flipud(V[i, :, :]) if V is not None else None,
#                                    np.flipud(policy[i, :, :]) if policy is not None else None]}],
#                     "label": str(i)
#                 } for i in range(index_min, index_max + 1)
#             ]
#         }]
#     )
#
#     return fig
#plot_V_policy(V_MC, policy_MC, index_min=1, index_max=4)