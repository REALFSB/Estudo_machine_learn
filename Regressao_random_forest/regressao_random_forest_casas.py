import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Carregando os dados de preços de casas a partir de um arquivo CSV
base_casas = pd.read_csv('../Dados/house_prices.csv')

# Exibindo os dados carregados
print(base_casas)

# Exibindo as estatísticas descritivas dos dados
print(base_casas.describe())

# Extraindo a variável independente (coluna 5) e a variável dependente (coluna 2)
x_casas = base_casas.iloc[:, 5:6].values
y_casas = base_casas.iloc[:, 2].values

# Dividindo os dados em conjuntos de treinamento (70%) e teste (30%)
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0)

# Criação do modelo de regressão de floresta aleatória com 100 árvores
regressor_random_forest_casas = RandomForestRegressor(n_estimators=100)

# Treinamento do modelo com os dados de treinamento
regressor_random_forest_casas.fit(x_casas_treinamento, y_casas_treinamento)

# Avaliação do modelo usando os dados de teste
print(regressor_random_forest_casas.score(x_casas_teste, y_casas_teste))

# Previsão dos preços das casas usando os dados de teste
previsao = regressor_random_forest_casas.predict(x_casas_teste)
print(previsao)
