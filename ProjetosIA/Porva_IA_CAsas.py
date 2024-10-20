
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  # Para dividir os dados em conjuntos de treino e teste
from sklearn.datasets import fetch_california_housing  # Para carregar o conjunto de dados de habitação da Califórnia

# Carregar o dataset de California Housing
housing = fetch_california_housing()  # Carrega os dados de preços de casas na Califórnia
data = pd.DataFrame(housing.data, columns=housing.feature_names)  # Cria um DataFrame com os dados
data['PRICE'] = housing.target  # Adiciona uma coluna com os preços das casas ao DataFrame

# Exibir as primeiras linhas do dataset para entender a estrutura dos dados
print("Visualizando as primeiras linhas do dataset:")
print(data.head())  


X = data[['AveRooms']]  # Seleciona a coluna 'AveRooms' (número médio de quartos) 
y = data['PRICE']  # Seleciona a coluna 'PRICE' como a variável dependente

# Dividir o dataset em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e ajustar o modelo de regressão linear
model = LinearRegression()  
model.fit(X_train, y_train)  # Treina o modelo usando os dados de treino

# Fazer previsões com o modelo
predictions = model.predict(X_test)  # Usa o modelo treinado para prever os preços das casas no conjunto de teste

# Separar os dados em listas com base no número médio de quartos
precos_por_quartos = {}

# Preencher o dicionário com listas
for index, row in data.iterrows():
    quartos = row['AveRooms']  # Número médio de quartos
    preco = row['PRICE']  # Preço da casa

    # Verifica se o número de quartos já é uma chave no dicionário
    if quartos not in precos_por_quartos:
        precos_por_quartos[quartos] = []  # Cria uma nova lista se a chave não existir
    
    
    precos_por_quartos[quartos].append(preco)

# Exibir os preços separados por número de quartos
print("\nPreços das casas separados por número médio de quartos:")
for quartos, precos in precos_por_quartos.items():
    print(f"Número de Quartos: {quartos}, Preços: {precos}")

# Solicitar ao usuário um novo valor para o número médio de quartos
novo_valor_input = input("\nInsira o número médio de quartos (ex: 6): ")
novo_valor = np.array([[float(novo_valor_input)]])  # Converte a entrada do usuário em um formato numérico 2D

# Fazer a previsão com o novo valor inserido pelo usuário
y_pred = model.predict(novo_valor) 

print(f"\nPara o número médio de quartos {novo_valor[0][0]}, o valor previsto da casa é ${y_pred[0] * 1000:.2f}")

# Visualizar o gráfico das previsões
plt.scatter(X_test, y_test, color='red', alpha=0.5, label='Dados reais')  
plt.scatter(X_test, predictions, color='blue', alpha=0.5, label='Previsões')  

# Traçar a linha da regressão
plt.plot(X_test, predictions, color='green', linestyle='--', label='Linha de Regressão')  # Plota a linha de regressão

# Configurações do gráfico
plt.rcParams['figure.figsize'] = [10, 6]  
plt.title('Regressão do Preço das Casas na Califórnia')  
plt.xlabel('Número Médio de Quartos')  
plt.ylabel('Preço da Casa (em milhares)')  
plt.legend()  
plt.show()  
