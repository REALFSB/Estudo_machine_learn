import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

base_casas = pd.read_csv('../Dados/house_prices.csv')

print(base_casas)
print(base_casas.describe())

# Extraindo a variável independente (coluna 5) e a variável dependente (coluna 2)
x_casas = base_casas.iloc[:, 5:6].values
y_casas = base_casas.iloc[:, 2].values

# Dividindo os dados em conjuntos de treinamento (70%) e teste (30%)
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0)

# Cria e treina o modelo de regressão com árvore de decisão
regressor_arvore_casas = DecisionTreeRegressor()
regressor_arvore_casas.fit(x_casas_treinamento, y_casas_treinamento)

# Exibindo o coeficiente de determinação (R²) para os dados de treinamento e teste
print(regressor_arvore_casas.score(x_casas_treinamento, y_casas_treinamento))
print(regressor_arvore_casas.score(x_casas_teste, y_casas_teste))

# Faz previsões com os dados de teste
previsoes = regressor_arvore_casas.predict(x_casas_teste)

# Calculando e exibindo o erro médio absoluto das previsões
print(mean_absolute_error(y_casas_teste, previsoes))
