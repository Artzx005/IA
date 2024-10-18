from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Carregar dataset
iris = datasets.load_iris() 
features = iris.data
forgets = iris.target

lista_featuresAll = []
lista_featuresAll2 = []

# Preenchendo listas com as características desejadas
for obs in features:
    lista_featuresAll.append([obs[2]])  # precisa ser uma lista dentro da lista para ter formato 2D
    lista_featuresAll2.append([obs[3]])

# Criar e ajustar o modelo de regressão linear
model = LinearRegression()
model.fit(lista_featuresAll, lista_featuresAll2)  # Ajuste o modelo

# Fazer previsões
predictions = model.predict(lista_featuresAll)

# Testar com um novo valor
novo_valor = [[10]]
y_pred = model.predict(novo_valor)

# Exibir o resultado
print(f"\nPara o valor {novo_valor[0][0]}, o valor previsto de y é {y_pred[0]}")

# Visualizar o gráfico
plt.scatter(lista_featuresAll, lista_featuresAll2, color='red', alpha=0.5)
plt.plot(lista_featuresAll, predictions, color='blue', label='Regressão Linear')
plt.rcParams['figure.figsize'] = [10, 8]
plt.title('Iris Dataset Scatter Plot')
plt.xlabel('Petal Width')
plt.ylabel('Petal Height')
plt.legend()
plt.show()
