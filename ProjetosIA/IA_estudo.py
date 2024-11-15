import pandas as pd
#data = [{'a': 1, 'b': 2}, {'a':5, 'b':10, 'c': 20}]
#df = pd.DataFrame(data, index = ['primeiro', 'segundo'])
#print(df)

#data = [{'a':1, 'b': 2 }, {'a':5 ,'b': 10, 'c':20}]

#df1 = pd.DataFrame(data, index = ['primeiro', 'segundo'],
                   #columns= ['a', 'b'])

#df2 = pd.DataFrame(data, index= ['priemiro', 'segundo'],
                   #columns=['a', 'b1'])
#print(df1)
#print(df2)

#dic = {'um' : pd.Series([1, 2, 3], index= ['a', 'b', 'c']),
      # 'dois': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
#df = pd.DataFrame(dic)
#print(df['um'])
#print('Adicioando uma nova coluna passando como Série:')
#df['tres'] = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
#print(df)
#df['quatro'] = df['um'] + df['tres']
#print(df)
#print('Excluindo a coluna numero 1 com função del:')
#del df['um']
#print(df)
#print('Excluindo a priemira coluna com o POP')
#df.pop('dois')
#print(df)
#As linhas podem ser selecionadas passando o rótulo da linha
#print(df.loc['b'])
#Selecção por localização numerica inteira
#print(df.iloc[2])
#fatias de linhas ( slice)
#print(df[2:4])
#adicionando novas linhas a um DataFrame
df = pd.DataFrame([[1,2], [3,4]], columns=['a', 'b'])
df2 = pd.DataFrame([[5, 6], [7,8]], columns=['a', 'b'])
df = pd.concat([df, df2], ignore_index=True)
df = df.drop(1)
print(df)

