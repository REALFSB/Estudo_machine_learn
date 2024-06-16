import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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

# Escalando as variáveis independentes e dependentes
scaler_x_casas = StandardScaler()
x_casas_treinamento_scaled = scaler_x_casas.fit_transform(x_casas_treinamento)
x_casas_teste_scaled = scaler_x_casas.transform(x_casas_teste)

scaler_y_casas = StandardScaler()
y_casas_treinamento_scaled = scaler_y_casas.fit_transform(y_casas_treinamento.reshape(-1, 1))
y_casas_teste_scaled = scaler_y_casas.transform(y_casas_teste.reshape(-1, 1))

# Treinando o modelo SVR com kernel RBF
regressor_svr_casas = SVR(kernel='rbf')
regressor_svr_casas.fit(x_casas_treinamento_scaled, y_casas_treinamento_scaled.ravel())

# Avaliando o modelo
print("R^2 treinamento:", regressor_svr_casas.score(x_casas_treinamento_scaled, y_casas_treinamento_scaled))
print("R^2 teste:", regressor_svr_casas.score(x_casas_teste_scaled, y_casas_teste_scaled))

# Fazendo previsões no conjunto de teste
previsoes_scaled = regressor_svr_casas.predict(x_casas_teste_scaled)

# Invertendo a escala das previsões e dos valores reais
previsoes_inverse = scaler_y_casas.inverse_transform(previsoes_scaled.reshape(-1, 1))
y_casas_teste_inverse = scaler_y_casas.inverse_transform(y_casas_teste_scaled)

# Exibindo os valores reais e as previsões
print("Valores reais:", y_casas_teste_inverse.ravel())
print("Previsões:", previsoes_inverse.ravel())

# Calculando o erro absoluto médio
erro_absoluto_medio = mean_absolute_error(y_casas_teste_inverse, previsoes_inverse)
print("Mean Absolute Error:", erro_absoluto_medio)
