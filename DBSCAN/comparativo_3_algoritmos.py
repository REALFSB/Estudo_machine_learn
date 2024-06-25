from sklearn import datasets
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# Gerando dados de exemplo em forma de lua crescente
x_random, y_random = datasets.make_moons(n_samples=1500, noise=0.05)

# Aplicando KMeans com 2 clusters
kmeans = KMeans(n_clusters=2)
rotulos = kmeans.fit_predict(x_random)

# Exibindo gráfico KMeans
grafico = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=rotulos)
grafico.show()

# Aplicando AgglomerativeClustering com 2 clusters
hc = AgglomerativeClustering(n_clusters=2, linkage='ward')
rotulos = hc.fit_predict(x_random)

# Exibindo gráfico AgglomerativeClustering
grafico1 = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=rotulos)
grafico1.show()

# Aplicando DBSCAN com epsilon 0.1
dbscan = DBSCAN(eps=0.1)
rotulos = dbscan.fit_predict(x_random)

# Exibindo gráfico DBSCAN
grafico2 = px.scatter(x=x_random[:, 0], y=x_random[:, 1], color=rotulos)
grafico2.show()
