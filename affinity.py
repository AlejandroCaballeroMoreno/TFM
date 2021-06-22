import sys
from matplotlib.transforms import Bbox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from pandas_datareader import data
from sklearn import cluster, covariance, manifold
print(__doc__)

empresas = {
    'Aon' : 'AON',
    'Willis Towers Watson' : 'WLTW',
    'Marsh' : 'MMC',
    'JPMorgan' : 'JPM',
    'Visa' : 'V',
    'Bank of America' : 'BAC',
    'Mastercard' : 'MA',
}

fuente_datos = 'yahoo'  # Fuente de datos -> Yahoo Finance
fecha_inicio = '2015-01-01'
fecha_fin = '2021-06-04'
df = data.DataReader(list(empresas.values()), fuente_datos, fecha_inicio, fecha_fin)

symbols, names = np.array(sorted(empresas.items())).T

print(symbols)
print(names)

stock_open = np.array(df['Open']).T  # stock_open es un vector de la transpuesta de df['Open']
stock_close = np.array(df['Close']).T  # stock_close es un vector de la transpuesta de df['Close']

movimientos = stock_close - stock_open

# Aprendemos una estructura grafica de las correlaciones
modelo = covariance.GraphicalLassoCV()

# Usar correlaciones en vez de la covarianza es mas eficiente para la recuperación de estructuras
X = movimientos.copy().T  
X /= X.std(axis = 0)
modelo.fit(X)

# Cluster usando affinity propagation
_, labels = cluster.affinity_propagation(modelo.covariance_,random_state=0)
n_labels = labels.max()

for i in range (n_labels + 1):
    print('Cluster %i: %s' %((i+1), ', '.join(names[labels == i])))

# Encontrar una "Low-dimension embedding" para la visualizacion: encontrar la mejor posicion de los nodos (los stocks) en un plano de dos dimensiones
# Se usará un eigensolver denso para conseguir reproducibilidad
# Usaremos un numero alto de grupos para conseguir una estructura de larga escala
modelo_posicion_nodos = manifold.LocallyLinearEmbedding(n_components=2, eigen_solver='dense', n_neighbors=6)
embedding = modelo_posicion_nodos.fit_transform(X.T).T

# Visualizacion
plt.figure(1, facecolor='w', figsize=(10,8))
plt.clf()
ax = plt.axes([0.,0.,1.,1.])
plt.axis('off')

# Mostrar una grafica de las correlaciones parciales
correlaciones_parciales = modelo.precision_.copy()
d = 1/np.sqrt(np.diag(correlaciones_parciales))
correlaciones_parciales *= d
correlaciones_parciales *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(correlaciones_parciales, k=1))>0.02)

# Dibujar los nodos usando las coordenadas de nuestro embedding
plt.scatter(embedding[0], embedding[1], s = 100*d**2, c=labels,cmap=plt.cm.nipy_spectral)

# Dibujamos los límites
start_idx, end_idx = np.where(non_zero)
# Una secuencia de (*linea0* *linea1* *linea2*) donde:
# linean = (x0,y0),(x1,y1),...,(xm,ym)
segmentos = [[embedding[:,start], embedding[:,stop]] for start, stop in zip(start_idx, end_idx)]
valores = np.abs(correlaciones_parciales[non_zero])
lc = LineCollection(segmentos,
zorder = 0, cmap = plt.cm.hot_r, norm = plt.Normalize(0, .7*valores.max()))
lc.set_array(valores)
lc.set_linewidths(15*valores)
ax.add_collection(lc)

# Añadir una etiqueta a cada nodo. El reto aquí es que queremos posiciones las etiquetas
# de forma que se evite el overlap con otras etiquetas.
for index, (name, label,(x,y)) in enumerate (zip(names, labels, embedding.T)):
    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]

    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002

    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002

    plt.text(x,y,name,size=10,
    horizontalalignment=horizontalalignment,
    verticalalignment=verticalalignment,
    bbox=dict(facecolor='w',
    edgecolor=plt.cm.nipy_spectral(label/float(n_labels)),
    alpha = .6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())

plt.show()

    
