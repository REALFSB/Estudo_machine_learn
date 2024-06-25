import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.cluster import DBSCAN

# Definindo listas de valores para x (idades) e y (salários)
x = [20, 27, 21, 25, 30, 32, 28, 24, 26, 29, 23, 22, 31, 19, 33]
y = [1000, 1200, 1900, 1500, 1700, 2000, 2100, 1800, 1600, 1750, 1600, 1400, 1950, 1100, 2050]

# Combinando as listas x e y em um array bidimensional
base_salario = np.array(list(zip(x, y)))

# Normalização:
scaler_salario = StandardScaler()

# Normalizando os dados de entrada
base_salario = scaler_salario.fit_transform(base_salario)

# Aplicando o algoritmo DBSCAN:
dbscan_salario = DBSCAN(eps=0.5, min_samples=2)

# Ajustando o modelo aos dados normalizados
dbscan_salario.fit(base_salario)

# Obtendo os rótulos dos clusters identificados pelo DBSCAN
rotulos = dbscan_salario.labels_

# Criando e exibindo um gráfico de dispersão interativo:
grafico = px.scatter(x=base_salario[:, 0], y=base_salario[:, 1], color=rotulos)
grafico.show()
