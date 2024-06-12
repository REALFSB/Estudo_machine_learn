import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Carrega os dados previamente processados a partir de um arquivo pickle
with open('../Dados/credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Concatena os dados de treinamento e teste para formar um único conjunto de dados
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)

# Treina um classificador de rede neural
classficador_rede = MLPClassifier(activation='relu',   # Função de ativação ReLU
                                  batch_size=56,       # Tamanho do lote
                                  solver='adam')       # Otimizador Adam
classficador_rede.fit(x_credit, y_credit)              # Ajusta o classificador aos dados

# Treina um classificador de árvore de decisão
classficador_arvore = DecisionTreeClassifier(criterion='entropy',   # Critério de entropia
                                             min_samples_leaf=1,    # Número mínimo de amostras por folha
                                             min_samples_split=5,   # Número mínimo de amostras para dividir um nó
                                             splitter='best')       # Estratégia de divisão 'best'
classficador_arvore.fit(x_credit, y_credit)                        # Ajusta o classificador aos dados

# Treina um classificador SVM (Máquina de Vetores de Suporte)
classficador_svm = SVC(C=2.0,      # Parâmetro de regularização
                       kernel='rbf')  # Tipo de kernel RBF (Radial Basis Function)
classficador_svm.fit(x_credit, y_credit)  # Ajusta o classificador aos dados

# Salva os classificadores treinados em arquivos usando pickle
pickle.dump(classficador_rede, open('../Dados/rede_neural_finalizado.sav', 'wb'))
pickle.dump(classficador_arvore, open('../Dados/arvore_finalizado.sav', 'wb'))
pickle.dump(classficador_svm, open('../Dados/svm_finalizado.sav', 'wb'))
