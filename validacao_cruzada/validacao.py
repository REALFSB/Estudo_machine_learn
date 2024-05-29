from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle

########################################################################################################################

with open('../credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

########################################################################################################################

x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)

########################################################################################################################

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

# Arvore de decisão
grid_search_arvore = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parametros_arvore)
grid_search_arvore.fit(x_credit, y_credit)
melhores_parametros_arvore = grid_search_arvore.best_params_
melhores_resultado_arvore = grid_search_arvore.best_score_

# Random Forest
grid_search_randomflorest = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros_randomflorest)
grid_search_randomflorest.fit(x_credit, y_credit)
melhores_parametros_randomflorest = grid_search_arvore.best_params_
melhores_resultado_randomflorest = grid_search_arvore.best_score_

# KNN
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros_knn)
grid_search_knn.fit(x_credit, y_credit)
melhores_parametros_knn = grid_search_arvore.best_params_
melhores_resultado_knn = grid_search_arvore.best_score_

# Regressão Logistica
grid_search_regressao = GridSearchCV(estimator=LogisticRegression(), param_grid=parametros_regressao)
grid_search_regressao.fit(x_credit, y_credit)
melhores_parametros_regressao = grid_search_arvore.best_params_
melhores_resultado_regressao = grid_search_arvore.best_score_

# SVM
grid_search_svc = GridSearchCV(estimator=SVC(), param_grid=parametros_svc)
grid_search_regressao.fit(x_credit, y_credit)
melhores_parametros_svc = grid_search_arvore.best_params_
melhores_resultado_svc = grid_search_arvore.best_score_

# Redes Neurais
grid_search_redes = GridSearchCV(estimator=MLPClassifier(), param_grid=parametros_redes)
grid_search_redes.fit(x_credit, y_credit)
melhores_parametros_redes = grid_search_arvore.best_params_
melhores_resultado_redes = grid_search_arvore.best_score_

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
