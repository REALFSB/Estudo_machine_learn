import pandas as pd
from scipy.stats import shapiro, f_oneway
from statsmodels.stats.multicomp import MultiComparison
import numpy as np
import pickle

########################################################################################################################

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
# confiabilidade do teste
alpha = 0.05

# Realizando o teste de Shapiro-Wilk
print(shapiro(arvore))
print(shapiro(knn))
print(shapiro(logistica))
print(shapiro(svn))
print(shapiro(rede))
print(shapiro(randomforest))

# Teste de hipótese com ANOVA e Tukey
_, p = f_oneway(arvore, knn, logistica, svn, rede, randomforest)

if p <= alpha:
    print("\nHipotese nula rejeitada. Dados diferentes")
else:
    print("\nHipotese alternativa rejeitada. Dados iguais")

resultados_algoritmo = {'accuracy': np.concatenate([arvore, knn, logistica, svn, rede, randomforest]),
                        'algoritmo': ['arvore', 'arvore', 'arvore', 'arvore', 'arvore', 'arvore', 'arvore', 'arvore',
                                      'arvore', 'arvore',
                                      'arvore', 'arvore', 'arvore', 'arvore', 'arvore', 'arvore', 'arvore', 'arvore',
                                      'arvore', 'arvore',
                                      'arvore', 'arvore', 'arvore', 'arvore', 'arvore', 'arvore', 'arvore', 'arvore',
                                      'arvore', 'arvore',
                                      'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn',
                                      'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn',
                                      'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn', 'knn',
                                      'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica',
                                      'logistica', 'logistica', 'logistica', 'logistica',
                                      'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica',
                                      'logistica', 'logistica', 'logistica', 'logistica',
                                      'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica',
                                      'logistica', 'logistica', 'logistica', 'logistica',
                                      'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn',
                                      'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn',
                                      'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn', 'svn',
                                      'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede',
                                      'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede',
                                      'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede', 'rede',
                                      'randomforest', 'randomforest', 'randomforest', 'randomforest', 'randomforest',
                                      'randomforest', 'randomforest', 'randomforest', 'randomforest', 'randomforest',
                                      'randomforest', 'randomforest', 'randomforest', 'randomforest', 'randomforest',
                                      'randomforest', 'randomforest', 'randomforest', 'randomforest', 'randomforest',
                                      'randomforest', 'randomforest', 'randomforest', 'randomforest', 'randomforest',
                                      'randomforest', 'randomforest', 'randomforest', 'randomforest', 'randomforest']}

resultados_df = pd.DataFrame(resultados_algoritmo)

compara_algoritmos = MultiComparison(resultados_df['accuracy'], resultados_df['algoritmo'])

# Compara todos os grupos
teste_estatistico = compara_algoritmos.tukeyhsd()

print(teste_estatistico)