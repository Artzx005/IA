from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

iris = datasets.load_iris()
features = iris.data
target = iris.target

featuresAll = []

for obs in features:
    featuresAll.append([ obs [0] + obs[1] + obs[2] + obs[3]])
    
model = LinearRegression

model.fit(featuresAll, target)

predictions = model.predict(featuresAll)

plt.scatter(featuresAll, target, color  ='red', alpha=1.0)
plt.plot(featuresAll, predictions, color = 'Blue', label = 'Regressão Linear')
plt.rcParams['figure.figsize'] = [10,8]
plt.title('Iris Dataset scartter Plot')
plt.xlabel('Petal Width')
plt.ylabel('Petal height')
plt.show()