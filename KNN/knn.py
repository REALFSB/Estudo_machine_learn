from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pickle

########################################################################################################################

with open('../Dados/credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

with open('../Dados/census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

########################################################################################################################
# Criar e treinar os classificadores KNN
knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_credit.fit(x_credit_treinamento, y_credit_treinamento)

knn_census = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_census.fit(x_census_treinamento, y_census_treinamento)

########################################################################################################################
# Fazer previsões
previsao_credit = knn_credit.predict(x_credit_teste)
previsao_census = knn_census.predict(x_census_teste)

########################################################################################################################
# Avaliar os resultados
credit_matriz = classification_report(y_credit_teste, previsao_credit)
census_matriz = classification_report(y_census_teste, previsao_census)

########################################################################################################################
# Exibir os resultados
print(credit_matriz)
print(census_matriz)
