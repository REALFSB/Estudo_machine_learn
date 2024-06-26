# Repositório de estudos de Machine Learning

Este repositório contém códigos e exemplos que eu uso para estudos de Machine Learning, abrangendo várias técnicas e algoritmos. O foco é em pré-processamento de dados, diferentes métodos de aprendizagem e avaliação de algoritmos de classificação, utilizando bibliotecas populares como Scikit-learn, pandas, numpy, pickle, matplotlib, seaborn e plotly.

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Study%20Repository-blue)
![Data Science](https://img.shields.io/badge/Data%20Science-Study%20Repository-blue)
![Data Analysis](https://img.shields.io/badge/Data%20Analysis-Study%20Repository-blue)

## Conteúdo

**1-  Pré-processamento de dados**
- Limpeza de Dados
  - Tratamento de valores inconsistentes
  - Tratamento de valores faltantes
- Transformação de Dados
  - Divisão entre previsores e classe
  - Escalonamento dos atributos
  - Atributos categóricos - LabelEncoder
  - Atributos categóricos - OneHotEncoder
- Visualização dos Dados
  - Visualização inicial dos dados
  - Gráficos e estatísticas descritivas
- Normalização e Padronização
- Divisão de Dados
  - Bases de treinamento e teste
- Salvar as Bases de Dados
  - Utilização de Pickle para salvar e carregar dados pré-processados

**2- Algoritmos de Aprendizagem**

- Aprendizagem Bayesiana
- Árvores de Decisão
- Random forest
- Aprendizagem por Regras
- Aprendizagem Baseada em Instâncias
- Regressão Logística
- Máquinas de Vetores de Suporte (SVM)
- Redes Neurais Artificiai

**3- Avaliação de Algoritmos**

- Matrizes de confusão
- Verdadeiro Positivo e Falso Positivo
- Precision e Recall
- Overfitting e Underfitting
- Validação Cruzada
- Tuning dos Parâmetros
- Variância, Desvio Padrão, Coeficiente de Variação, Distribuição Normal, Teste de Hipóteses
- Combinação de Classificadores
- Rejeição de Classificadores

**4- Regressão da Aprendizagem de Maquina**

- Conceitos básicos sobre correlação
- Regressão linear (simples e múltipla)
- Regressão polinomial
- Regressão com árvores de decisão e random forest
- Regressão com vetores de suporte (SVR)
- Regressão com redes neurais artificiais

**5- Regras de Associação e Algoritmos Associativos**

- Introdução teórica sobre as regras de associação e suas principais aplicações
- Funcionamento do algoritmo Apriori
- Funcionamento do algoritmo ECLAT

**6- Agrupamento (Clustering)**

- K-means
- Agrupamento Hierárquico
  - Dendrogramas
- DBSCAN
  - Escolha dos parâmetros (epsilon e min_samples)

**7- Técnicas Avançadas em Machine Learning**
- Aprendizagem por Reforço
- Processamento de Linguagem Natural
- Visão Computacional
- Tratamento de Dados Desbalanceados
- Seleção de Atributos
- Redução de Dimensionalidade (PCA, LDA, Kernel PCA)
- Detecção de Outliers (utilizada para detecção de fraudes)
- Séries Temporais

## Dependências Principais

- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [Pickle](https://docs.python.org/3/library/pickle.html)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Plotly](https://plotly.com/)
- [SciPy](https://plotly.com/)
- Gym
- IPython

## Dependências de Modelagem e Aprendizado de Máquina

- DecisionTreeClassifier, DecisionTreeRegressor
- RandomForestClassifier, RandomForestRegressor
- KMeans, AgglomerativeClustering, DBSCAN
- MLPClassifier, MLPRegressor
- LinearRegression, LogisticRegression
- SVR, SVC
- KNeighborsClassifier

## Outras Dependências Específicas

- Apriori
- ECLAT
- StatsModels
- OpenCV

## Como Utilizar
1. Clone o repositório para sua máquina local:
`git clone https://github.com/REALFSB/Estudo_machine_learn.git`

2. Navegue até o diretório do repositório:
`cd Estudo_machine_learn` 

3. Instale as dependências:
- Scikit-learn: 
    `pip install scikit-learn`
- Pandas:
    `pip install pandas`
- Numpy:
    `pip install numpy`
- Matplotlib: 
    `pip install matplotlib`
- Seaborn:
    `pip install seaborn`
- Plotly:
    `pip install plotly`
- Apyori:
    `pip install apyori`
- ECLAT:
    `pip install pyECLAT`
- Gym:
    `pip install gym`
- IPython:
    `pip install IPython`
- Cv2:
    `pip install opencv-contrib-python`