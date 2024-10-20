import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Função para consultar o preço do veículo na Tabela FIPE para diferentes anos
def consultar_preco_fipe(codigo_fipe):
    url = f"https://brasilapi.com.br/api/fipe/preco/v1/{codigo_fipe}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()  # Retorna os dados da API
    else:
        print(f"Erro ao acessar a API: {response.status_code}")
        return None

# Função para converter o valor de string para float
def converter_valor(valor_str):
    # Remover "R$" e pontos, e trocar vírgula por ponto
    valor_str = valor_str.replace('R$', '').replace('.', '').replace(',', '.').strip()
    return float(valor_str)

# Lista com códigos FIPE de Kombi em diferentes anos
codigos_fipe = [
    "005238-8",  # Kombi em um ano específico (trocar conforme necessário)
    "005024-5",  # Outro código FIPE para Kombi de outro ano
    # Adicionar mais códigos conforme necessário
]

# Listas para armazenar anos e valores
anos = []
valores = []

# Fazer requisições à API para todos os códigos FIPE (antigo e novo)
for codigo in codigos_fipe:
    data = consultar_preco_fipe(codigo)
    if data:
        # Adicionar os anos e converter os valores à lista
        anos.append([data[0]['anoModelo']])  # API retorna uma lista de dicionários
        valores.append([converter_valor(data[0]['valor'])])  # Converter para float

# Converter listas para DataFrame (opcional para visualização)
df = pd.DataFrame({'Ano': [ano[0] for ano in anos], 'Valor': [valor[0] for valor in valores]})

# Criar e ajustar o modelo de regressão linear
model = LinearRegression()
model.fit(anos, valores)  # Ajustar o modelo com dados reais

# Fazer previsões
predictions = model.predict(anos)

# Testar com um novo ano (exemplo: prever o valor de uma Kombi em 2025)
ano = int(input("digite o ano da previsão:"))
novo_ano = [[ano]]
y_pred = model.predict(novo_ano)
print(f"\nPara o ano {novo_ano[0][0]}, o valor previsto é R$ {y_pred[0][0]:.2f}")

# Plotar gráfico de dispersão e linha de regressão
plt.scatter(anos, valores, color='red', alpha=0.5, label='Dados Reais')
plt.plot(anos, predictions, color='blue', label='Regressão Linear')
plt.rcParams['figure.figsize'] = [10, 8]
plt.title('Regressão Linear - Preço de Kombis ao Longo dos Anos (Dados Reais)')
plt.xlabel('Ano')
plt.ylabel('Valor (R$)')
plt.legend()
plt.show()

# Exibir dados no DataFrame
print(df)
