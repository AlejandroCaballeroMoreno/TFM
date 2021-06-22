from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.pylab import *



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

stock_open = np.array(df['Open']).T  # stock_open es un vector de la transpuesta de df['Open']
stock_close = np.array(df['Close']).T  # stock_close es un vector de la transpuesta de df['Close']

movimientos = stock_close - stock_open
suma_movimientos = np.sum(movimientos, 1)

for i in range (len(empresas)):
   print('Empresa: {}, Cambio: {}'.format(df['High'].columns[i], suma_movimientos[i]))

# Visualizar los datos:
plt.figure(figsize = (20,10))
plt.subplot(1,2,1)
plt.title('Empresa : Aon', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Opening Price', fontsize = 15)
plt.plot(df['Open']['AON'])
plt.grid()
plt.subplot(1,2,2)
plt.title('Empresa: Marsh', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Opening Price', fontsize = 15)
plt.plot(df['Open']['MMC'])
plt.grid()
plt.show()

# Visualización de datos: valor de apertura y cierre en los últimos 30 dias
plt.figure(figsize = (20,10))
plt.title('Empresa : Aon', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.xlabel('Date', fontsize = 20)
plt.ylabel('Price', fontsize = 20)
plt.plot(df.iloc[1580:-1]['Open']['AON'], label = 'Open') # Valores de apertura de los últimos 30 dias frente a fecha
plt.plot(df.iloc[1580:-1]['Close']['AON'], label = 'Close') # Valores de cierre de los últimos 30 dias frente a fecha
plt.legend(loc = 'upper left', frameon = False, framealpha = 1, prop = {'size': 22}) # Propiedades de la caja de leyendas
plt.grid()
plt.show()

#print(movimientos.shape)
# Visualización de los movimientos
plt.figure(figsize = (20,8))
plt.title('Empresa: Aon', fontsize = 20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 20)
plt.xlabel('Date', fontsize = 20)
plt.ylabel('Movimientos', fontsize = 20)
plt.plot(movimientos[0][1580:-1])
plt.grid()
plt.show()

# Visualización del volumen
plt.figure(figsize = (20,10))
plt.title('Empresa : Aon', fontsize = 20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 20)
plt.xlabel('Date', fontsize = 20)
plt.ylabel('Volumen', fontsize = 20)
plt.plot(df['Volume']['AON'], label = 'Volume')
plt.grid()
plt.show()

# Visualizacion de los movimientos del mercado con un CandleStick
fig = go.Figure(data  = [go.Candlestick(x = df.index,
open = df.iloc[1560:-1]['Open']['AON'],
high = df.iloc[1560:-1]['High']['AON'],
low = df.iloc[1560:-1]['Low']['AON'],
close = df.iloc[1560:-1]['Close']['AON'])])
fig.show()

# Need for normalization
# Visualization of variation of movement
plt.figure(figsize = (20,8))
ax1 = plt.subplot(1,2,1)
plt.title('Empresa: Aon', fontsize = 20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 20)
plt.xlabel('Date', fontsize = 20)
plt.ylabel('Movimientos', fontsize = 20)
plt.plot(movimientos[0])
plt.grid()
plt.subplot(1,2,2,sharey = ax1)
plt.title('Empresa: Willis Towers Watson', fontsize = 20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 20)
plt.xlabel('Date', fontsize = 20)
plt.ylabel('Movimientos', fontsize = 20)
plt.plot(movimientos[1])
plt.grid()
plt.show()

###########################################################################
normalizer = Normalizer()  # Se define el Normalizer

# Creamos un modelo de clustering KMeans
kmeans = KMeans(n_clusters=3, max_iter=1000)

# Creamos una pipeline que relacione el normalizer y el kmeans
pipeline = make_pipeline(normalizer, kmeans)

# Ajustamos la pipeline al movimiento diario de stocks
pipeline.fit(movimientos)
labels = pipeline.predict(movimientos)

df1 = pd.DataFrame({'Etiquetas': labels, 'Empresas':list(empresas.keys())}).sort_values(by=['Etiquetas'], axis = 0)
print(df1)

# PCA-Reduction
# Definimos otro normalizer
normalizer2 = Normalizer()

# Reducimos los datos
datos_reducidos = PCA(n_components = 2)

# Creamos el modelo KMeans
kmeans2 = KMeans(n_clusters=3, max_iter=1000)

# Creamos otra pipeline que relacione el normalizer, el PCA y el kmeans
pipeline2 = make_pipeline(normalizer2, datos_reducidos, kmeans2)

# Ajustamos la pipeline al movimiento diario de stocks
pipeline2.fit(movimientos)

# Predicción
labels2 = pipeline2.predict(movimientos)

# Creamos un dataframe para almacenar las empresas y las etiquetas que se han predicho
df2 = pd.DataFrame({'Etiquetas': labels2, 'Empresas':list(empresas.keys())}).sort_values(by=['Etiquetas'], axis = 0)
print(df2)

# Dibujamos los limites de decision
normalizer3 = Normalizer()
norm_movements = normalizer3.fit_transform(movimientos)
datos_reducidos2 = PCA(n_components=2).fit_transform(norm_movements)

# Definimos el tamaño de la red
h = 0.01

x_min, x_max = datos_reducidos2[:,0].min()-1, datos_reducidos2[:,0].max()+1
y_min, y_max = datos_reducidos2[:,1].min()-1, datos_reducidos2[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtenemos las etiquetas para cada punto en la red usando el modelo que hemos entrenado
Z = kmeans2.predict(np.c_[xx.ravel(), yy.ravel()])

# Dibujar el resultado

Z = Z.reshape(xx.shape)

# Definimos el dibujo
cmap = plt.cm.Paired

plt.clf()
plt.figure(figsize = (10,10))
plt.imshow(Z, cmap ,interpolation = 'nearest', extent = (xx.min(), xx.max(), yy.min(), yy.max()), aspect = 'auto', origin = 'lower')
plt.plot(datos_reducidos2[:,0], datos_reducidos2[:,1], 'k.', markersize = 5)

# Dibujamos el centroide de cada cluster
centroides = kmeans2.cluster_centers_
plt.scatter(centroides[:,0], centroides[:,1], marker = 'x', s = 169, linewidths=3, color = 'w', zorder=10)
plt.title('Clustering KMeans')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

