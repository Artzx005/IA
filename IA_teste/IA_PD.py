import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
features = iris.data
forgets = iris.target


df_iris = pd.DataFrame(np.column_stack((features, forgets)), columns = iris.feature_names + ['target'])
print(df_iris.describe())
