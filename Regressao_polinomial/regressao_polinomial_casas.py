import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

base_casas = pd.read_csv('../Dados/house_prices.csv')

print(base_casas)
print(base_casas.describe())

# Extraindo a variável independente (coluna 5) e a variável dependente (coluna 2)
x_casas = base_casas.iloc[:, 5:6].values
y_casas = base_casas.iloc[:, 2].values

# Dividindo os dados em conjuntos de treinamento (70%) e teste (30%)
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0)

# Transformação Polinomial:
poly = PolynomialFeatures(degree=2)
x_casas_treinamento_poly = poly.fit_transform(x_casas_treinamento)
x_casas_teste_poly = poly.transform(x_casas_teste)

# Criação e Treinamento do Modelo:
regressor_casas_poly = LinearRegression()
regressor_casas_poly.fit(x_casas_treinamento_poly, y_casas_treinamento)

# Avaliação do Modelo:
print(regressor_casas_poly.score(x_casas_treinamento_poly, y_casas_treinamento))
print(regressor_casas_poly.score(x_casas_teste_poly, y_casas_teste))

# Predições e Erro:
previsoes = regressor_casas_poly.predict(x_casas_teste_poly)
print(mean_absolute_error(y_casas_teste, previsoes))
