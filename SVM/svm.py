from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle

########################################################################################################################

with open("../Dados/credit.pkl", 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

with open("../Dados/census.pkl", 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

########################################################################################################################
# Treinamento do Modelo SVM:
svm_credit = SVC(kernel='rbf', random_state=1, C=2.0)
svm_credit.fit(x_credit_teste, y_credit_teste)

svm_census = SVC(kernel='linear', random_state=1, C=1.0)
svm_census.fit(x_census_treinamento, y_census_treinamento)

########################################################################################################################
# Previsões do Modelo:
previsao_credit = svm_credit.predict(x_credit_teste)

previsao_census = svm_census.predict(x_census_teste)

########################################################################################################################
# Avaliação dos Modelos:
matriz_credit = classification_report(y_credit_teste, previsao_credit)
presisao_credit = accuracy_score(y_credit_teste, previsao_credit)

matriz_census = classification_report(y_census_teste, previsao_census)
precisao_census = accuracy_score(y_census_teste, previsao_census)

########################################################################################################################
# Exibição dos Resultados:
print(matriz_credit)
print(presisao_credit)

print(matriz_census)
print(precisao_census)
