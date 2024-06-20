from sklearn.datasets import make_blobs
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# Geração de dados aleatórios:
x_random, y_random = make_blobs(n_samples=200, centers=5)

# Aplicação do algoritmo K-Means:
kmeans_blobs = KMeans(n_clusters=5)
kmeans_blobs.fit(x_random)

# Predição dos rótulos e obtenção dos centróides:
rotulos = kmeans_blobs.predict(x_random)
centroides = kmeans_blobs.cluster_centers_

# Visualização:
grafico = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=rotulos)
grafico2 = px.scatter(x=centroides[:, 0], y=centroides[:, 1], size=[12, 12, 12, 12, 12])
grafico3 = go.Figure(data=grafico.data + grafico2.data)
grafico3.show()
