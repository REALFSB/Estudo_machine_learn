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


grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parametros_arvore)
grid_search.fit(x_credit, y_credit)
melhores_parametros = grid_search.best_params_
melhores_resultado = grid_search.best_score_

print(melhores_parametros)
print(melhores_resultado)
