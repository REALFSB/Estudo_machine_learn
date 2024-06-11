import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

########################################################################################################################

with open('C:\\Users\\batis\\PycharmProjects\\MachineLearn\\risco_credito.pkl', 'rb') as f:
    x_risco_treinamento, y_risco_treinamento = pickle.load(f)

with open('C:\\Users\\batis\\PycharmProjects\\MachineLearn\\credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

with open('C:\\Users\\batis\\PycharmProjects\\MachineLearn\\census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

########################################################################################################################
# Pré-processamento dos Dados:
x_risco_treinamento = np.delete(x_risco_treinamento, [2, 7, 11], axis=0)
y_risco_treinamento = np.delete(y_risco_treinamento, [2, 7, 11], axis=0)

########################################################################################################################
# Treinamento dos Modelos de Regressão Logística:
risco = LogisticRegression(random_state=1)
risco.fit(x_risco_treinamento, y_risco_treinamento)

regressao_credit = LogisticRegression(random_state=1)
regressao_credit.fit(x_credit_treinamento, y_credit_treinamento)

regressao_census = LogisticRegression(random_state=1)
regressao_census.fit(x_census_treinamento, y_census_treinamento)

########################################################################################################################
# Realização de Previsões:
previsao_risco = risco.predict([[0, 0, 1, 2], [2, 0, 0, 0]])

previsao_credit = regressao_credit.predict(x_credit_teste)

previsao_census = regressao_census.predict(x_census_teste)

########################################################################################################################
# Avaliação dos Modelos:
matriz_credit = classification_report(y_credit_teste, previsao_credit)

matriz_census = classification_report(y_census_teste, previsao_census)

########################################################################################################################
# Exibição dos Resultados:
print(matriz_credit)
print(matriz_census)