import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objs as go

plano_saude = pd.read_csv('../Dados/plano_saude2.csv')

print(plano_saude)

x_plano_saude = plano_saude.iloc[:, 0:1].values
y_plano_saude = plano_saude.iloc[:, 1].values

# Transformação Polinomial:
poly = PolynomialFeatures(degree=2)
x_plano_saude_poly = poly.fit_transform(x_plano_saude)
print(x_plano_saude_poly)

# Criação e Treinamento do Modelo:
regressor_saude_polinomial = LinearRegression()
regressor_saude_polinomial.fit(x_plano_saude_poly, y_plano_saude)

# Exibição dos Coeficientes do Modelo:
print(regressor_saude_polinomial.intercept_)
print(regressor_saude_polinomial.coef_)

# Predição para um Novo Valor:
novo = [[40]]
novo = poly.transform(novo)
print(regressor_saude_polinomial.predict(novo))

# Predições e Visualização:
previoses = regressor_saude_polinomial.predict(x_plano_saude_poly)
print(previoses)
grafico = px.scatter(x=x_plano_saude[:, 0], y=y_plano_saude)
grafico.add_scatter(x=x_plano_saude[:, 0], y=previoses, name='Regressão')
grafico.show()