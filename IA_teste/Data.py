import numpy as np
import pandas as pd

#data = np.array(['a', 'b','c','d'])
#data2 = {'a' : 5, 'b' : 6, 'c' : 7, 'd' : 8,}
#s2 = pd.Series(data2)
#s1 = pd.Series(data)

#s = pd.Series([1, 2, 3, 4, 5], index = ['a', 'b', 'c', 'd', 'e']) estudo

#print(s[['a', 'b']])
#data = [1, 2, 3, 4 , 5] ficou para estudo
#data = [['Maria', 10],['Carlos', 11], ['Arthur', 12]]
data = {'Nome':['Marcos', 'Paula', 'Lia', 'Carlos'], 'Pontuação': [7.5, 8.0, 8.5,9.0]}
df = pd.DataFrame(data, index=['rank1', 'rank2', 'rank3', 'rank4'])
#df['Idade'] = df['Idade'].astype(float)
print(df)