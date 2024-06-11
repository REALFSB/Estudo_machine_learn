import pandas as pd
from scipy.stats import shapiro, f_oneway
from statsmodels.stats.multicomp import MultiComparison
import numpy as np
import pickle

########################################################################################################################
# Carregamento dos Resultados:
with open('resultado_arvore.pkl', 'rb') as f:
    arvore = pickle.load(f)

with open('resultados_knn.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('resultados_logistica.pkl', 'rb') as f:
    logistica = pickle.load(f)

with open('resultados_svn.pkl', 'rb') as f:
    svn = pickle.load(f)

with open('resultados_redeneural.pkl', 'rb') as f:
    rede = pickle.load(f)

with open('resultados_randomforest.pkl', 'rb') as f:
    randomforest = pickle.load(f)

########################################################################################################################
# Definição do Nível de Significância
alpha = 0.05

# Teste de Normalidade de Shapiro-Wilk:
print(shapiro(arvore))
print(shapiro(knn))
print(shapiro(logistica))
print(shapiro(svn))
print(shapiro(rede))
print(shapiro(randomforest))

########################################################################################################################
# Teste ANOVA:
_, p = f_oneway(arvore, knn, logistica, svn, rede, randomforest)

if p <= alpha:
    print("\nHipotese nula rejeitada. Dados diferentes")
else:
    print("\nHipotese alternativa rejeitada. Dados iguais")

########################################################################################################################
# Preparação dos Dados para o Teste de Tukey:
resultados_algoritmo = {'accuracy': np.concatenate([arvore, knn, logistica, svn, rede, randomforest]),
                        'algoritmo': ['arvore'] * 30 +
                                     ['knn'] * 30 +
                                     ['logistica'] * 30 +
                                     ['svn'] * 30 +
                                     ['rede'] * 30 +
                                     ['randomforest'] * 30}

resultados_df = pd.DataFrame(resultados_algoritmo)

########################################################################################################################
# Teste de Comparações Múltiplas de Tukey:
compara_algoritmos = MultiComparison(resultados_df['accuracy'], resultados_df['algoritmo'])
teste_estatistico = compara_algoritmos.tukeyhsd()

print(teste_estatistico)
