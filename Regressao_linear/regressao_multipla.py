import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Carregando a base de dados de preços de casas a partir de um arquivo CSV
base_casas = pd.read_csv('../Dados/house_prices.csv')

# Selecionando as variáveis independentes (features) e a variável dependente (target)
x_casas = base_casas.iloc[:, 3:19].values  # Variáveis independentes (colunas 3 a 18)
y_casas = base_casas.iloc[:, 2].values    # Variável dependente (coluna 2)

# Dividindo os dados em conjuntos de treinamento (70%) e teste (30%)
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0)

# Criando um modelo de regressão linear múltipla
regressor_multiplo_casas = LinearRegression()

# Ajustando o modelo aos dados de treinamento
regressor_multiplo_casas.fit(x_casas_treinamento, y_casas_treinamento)

# Imprimindo os coeficientes de regressão (coeficientes das variáveis independentes)
print(regressor_multiplo_casas.coef_)

# Avaliando o desempenho do modelo nos dados de treinamento e teste usando R²
print(regressor_multiplo_casas.score(x_casas_treinamento, y_casas_treinamento))
print(regressor_multiplo_casas.score(x_casas_teste, y_casas_teste))

# Realizando previsões com o modelo ajustado nos dados de teste
previsoes = regressor_multiplo_casas.predict(x_casas_teste)

# Calculando e imprimindo o Erro Absoluto Médio (MAE) entre as previsões e os valores reais
print(mean_absolute_error(y_casas_teste, previsoes))
