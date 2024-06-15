from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pickle

########################################################################################################################
# Carregamento dos Dados:
with open('../Dados/credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

########################################################################################################################
# Concatenando Dados de Treino e Teste:
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)

########################################################################################################################
# Definição dos Parâmetros para GridSearchCV:
parametros_arvore = {'criterion': ['gini', 'entropy'],
                     'splitter': ['best', 'random'],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 5, 10]}

parametros_randomflorest = {'criterion': ['gini', 'entropy'],
                            'n_estimators': [10, 40, 100, 150],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 5, 10]}

parametros_knn = {'n_neighbors': [3, 5, 10, 20],
                  'p': [1, 2]}

parametros_regressao = {'tol': [0.0001, 0.00001, 0.000001],
                        'C': [1.0, 1.5, 2.0],
                        'solver': ['lbfgs', 'sag', 'saga']}

parametros_svc = {'tol': [0.001, 0.0001, 0.00001],
                  'C': [1.0, 1.5, 2.0],
                  'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

parametros_redes = {'activation': ['relu', 'logistic', 'tahn'],
                    'solver': ['adam', 'sgd'],
                    'batch_size': [10, 56]}

########################################################################################################################
#GridSearchCV para Encontrar os Melhores Hiperparâmetros:

# Arvore de decisão
grid_search_arvore = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parametros_arvore)
grid_search_arvore.fit(x_credit, y_credit)
melhores_parametros_arvore = grid_search_arvore.best_params_
melhores_resultado_arvore = grid_search_arvore.best_score_

# Random Forest
grid_search_randomflorest = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros_randomflorest)
grid_search_randomflorest.fit(x_credit, y_credit)
melhores_parametros_randomflorest = grid_search_randomflorest.best_params_
melhores_resultado_randomflorest = grid_search_randomflorest.best_score_

# KNN
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros_knn)
grid_search_knn.fit(x_credit, y_credit)
melhores_parametros_knn = grid_search_knn.best_params_
melhores_resultado_knn = grid_search_knn.best_score_

# Regressão Logistica
grid_search_regressao = GridSearchCV(estimator=LogisticRegression(), param_grid=parametros_regressao)
grid_search_regressao.fit(x_credit, y_credit)
melhores_parametros_regressao = grid_search_regressao.best_params_
melhores_resultado_regressao = grid_search_regressao.best_score_

# SVM
grid_search_svc = GridSearchCV(estimator=SVC(), param_grid=parametros_svc)
grid_search_regressao.fit(x_credit, y_credit)
melhores_parametros_svc = grid_search_svc.best_params_
melhores_resultado_svc = grid_search_svc.best_score_

# Redes Neurais
grid_search_redes = GridSearchCV(estimator=MLPClassifier(), param_grid=parametros_redes)
grid_search_redes.fit(x_credit, y_credit)
melhores_parametros_redes = grid_search_redes.best_params_
melhores_resultado_redes = grid_search_redes.best_score_

########################################################################################################################
# Validação Cruzada e Armazenamento dos Resultados:
resultados_arvore = []
resultados_randomforest = []
resultados_knn = []
resultados_logistica = []
resultados_svn = []
resultados_redeneural = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    arvore = DecisionTreeClassifier(criterion='entropy',
                                    min_samples_leaf=1,
                                    min_samples_split=5,
                                    splitter='best')
    scores = cross_val_score(arvore, x_credit, y_credit, cv=kfold)

    resultados_arvore.append(scores.mean())

    #####################################################################

    random_forest = RandomForestClassifier(criterion='entropy',
                                           min_samples_leaf=1,
                                           min_samples_split=5,
                                           n_estimators=10)

    scores = cross_val_score(random_forest, x_credit, y_credit, cv=kfold)

    resultados_randomforest.append(scores.mean())

    #####################################################################

    knn = KNeighborsClassifier()
    scores = cross_val_score(knn, x_credit, y_credit, cv=kfold)

    resultados_knn.append(scores.mean())

    #####################################################################

    logistica = LogisticRegression(C = 1.0,
                                   solver='lbfgs',
                                   tol=0.0001)
    scores = cross_val_score(logistica, x_credit, y_credit, cv=kfold)

    resultados_logistica.append(scores.mean())

    #####################################################################

    svn = SVC(kernel='rbf',
              C = 2.0)
    scores = cross_val_score(svn, x_credit, y_credit, cv=kfold)

    resultados_svn.append((scores.mean()))

    #####################################################################

    rede_neural = MLPClassifier(activation='relu',
                                batch_size=56,
                                solver='adam')
    scores = cross_val_score(rede_neural, x_credit, y_credit, cv=kfold)

    resultados_redeneural.append(scores.mean())

########################################################################################################################

print(melhores_parametros_arvore)
print(melhores_resultado_arvore)

print(melhores_parametros_randomflorest)
print(melhores_resultado_randomflorest)

print(melhores_parametros_knn)
print(melhores_resultado_knn)

print(melhores_parametros_regressao)
print(melhores_resultado_regressao)

print(melhores_parametros_svc)
print(melhores_resultado_svc)

print(melhores_parametros_redes)
print(melhores_resultado_redes)

print("Testes realizados")

########################################################################################################################

with open('resultado_arvore.pkl', mode='wb') as f:
    pickle.dump(resultados_arvore, f)

with open('resultados_randomforest.pkl', mode='wb') as f:
    pickle.dump(resultados_randomforest, f)

with open('resultados_knn.pkl', mode='wb') as f:
    pickle.dump(resultados_knn, f)

with open('resultados_logistica.pkl', mode='wb') as f:
    pickle.dump(resultados_logistica, f)

with open('resultados_svn.pkl', mode='wb') as f:
    pickle.dump(resultados_svn, f)

with open('resultados_redeneural.pkl', mode='wb') as f:
    pickle.dump(resultados_redeneural, f)

########################################################################################################################

print("Codigo finalizado")
