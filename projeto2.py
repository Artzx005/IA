from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


iris = datasets.load_iris() 
features = iris.data
forgets = iris.target

lista_featuresAll = []
lista_featuresAll2 = []

for obs in features:
   # soma = features[0] + features[1] + features [2] + features[3]
    lista_featuresAll.append(obs[2])
    lista_featuresAll2.append(obs[3])
    
    
# iniciando    
model = LinearRegression()
#Ajustando o Modelo
model.fit (lista_featuresAll, lista_featuresAll2)
#Fazendo Previsões
predictions = model.predict(lista_featuresAll)

novo_valor = [[15]]
y_pred = model.predict(novo_valor)

print(f"/n para a soma das features {novo_valor[0] [0]}, o valor previsto de y é {y_pred[0]}")
#print(lista_featuresAll)
#print(lista_featuresAll2)
#     type (iris)
#     print(iris.data.shape)

plt.scatter(lista_featuresAll, lista_featuresAll2, color  ='red', alpha=0.5)
plt.plot(lista_featuresAll, predictions, color = 'Blue', label = 'Regressão Linear')
plt.rcParams['figure.figsize'] = [10,8]
plt.title('Iris Dataset scartter Plot')
plt.xlabel('Petal Width')
plt.ylabel('Petal height')
plt.show()