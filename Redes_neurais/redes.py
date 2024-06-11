import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

########################################################################################################################

with open('../Dados/credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

with open('../Dados/census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

########################################################################################################################
# Criação e Treinamento dos Modelos de Rede Neural:
rede_credit = MLPClassifier(max_iter=1500,
                            verbose=True,
                            tol=0.0000100,
                            solver='adam',
                            activation='relu',
                            hidden_layer_sizes=(2, 2))

rede_census = MLPClassifier(max_iter=1200,
                            verbose=True,
                            solver='adam',
                            activation='relu',
                            hidden_layer_sizes=(55, 55))

########################################################################################################################
# Previsões e Avaliação dos Modelos:
rede_credit.fit(x_credit_treinamento, y_credit_treinamento)
rede_census.fit(x_census_treinamento, y_census_treinamento)

previsao_credit = rede_credit.predict(x_credit_teste)
previsao_census = rede_census.predict(x_census_teste)

matriz_credit = classification_report(y_credit_teste, previsao_credit)
precisao_credit = accuracy_score(y_credit_teste, previsao_credit)

matriz_census = classification_report(y_census_teste, previsao_census)
precisao_census = accuracy_score(y_census_teste, previsao_census)

########################################################################################################################
# Exibição dos Resultados:
print(matriz_credit)
print(precisao_credit)

print(matriz_census)
print(precisao_census)
