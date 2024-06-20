import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Dados de Entrada:
x = [20, 27, 21, 25, 30, 32, 28, 24, 26, 29, 23, 22, 31, 19, 33]
y = [1000, 1200, 1900, 1500, 1700, 2000, 2100, 1800, 1600, 1750, 1600, 1400, 1950, 1100, 2050]

base_salario = np.array(list(zip(x, y)))

# Normalização:
scaler_salario = StandardScaler()
base_salario = scaler_salario.fit_transform(base_salario)

# Aplicação do K-means:
k_means_salario = KMeans(n_clusters=3)
k_means_salario.fit(base_salario)

# Saídas do K-means:
centroides = k_means_salario.cluster_centers_

print(centroides)
print(scaler_salario.inverse_transform(k_means_salario.cluster_centers_))

rotulos = k_means_salario.labels_
print(rotulos)

# Visualização:
grafico = px.scatter(x=base_salario[:, 0], y=base_salario[:, 1], color=rotulos)
grafico2 = px.scatter(x=centroides[:, 0], y=centroides[:, 1], size=[12, 12, 12])
grafico3 = go.Figure(data=grafico.data + grafico2.data)
grafico3.show()
