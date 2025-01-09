

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
