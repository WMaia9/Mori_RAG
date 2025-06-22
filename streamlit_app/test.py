import pandas as pd

# 1. Carregue o arquivo de metadados que o seu app está usando
#    (Ajuste o caminho se necessário)
caminho_metadados = "data/odas/metadados_odas_1606_v2.pkl"

try:
    df_odas = pd.read_pickle(caminho_metadados)
    print(f"Arquivo '{caminho_metadados}' carregado com sucesso!")
except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em '{caminho_metadados}'. Verifique o caminho.")
    exit()

# 2. Veja TODOS os valores únicos que realmente existem na sua coluna Rubrica_IA
print("\nValores únicos encontrados na coluna 'Rubrica_IA':")
if 'Rubrica_IA' in df_odas.columns:
    # O .dropna() remove os valores nulos (NaN) para a contagem não dar erro
    # O .unique() mostra cada valor diferente apenas uma vez
    valores_unicos = df_odas['Rubrica_IA'].dropna().unique()
    print(valores_unicos)
else:
    print("ERRO: A coluna 'Rubrica_IA' não foi encontrada no arquivo de metadados.")
    exit()

# 3. Defina a rubrica que estamos procurando, exatamente como o app faria
rubrica_que_buscamos = "Rubrica 2 - Exploração"
print(f"\nEstamos procurando por um texto que contenha: '{rubrica_que_buscamos}'")

# 4. Verifique se algum dos valores únicos contém o que buscamos
matches = [v for v in valores_unicos if rubrica_que_buscamos in str(v)]

if matches:
    print(f"\n✅ SUCESSO! Uma correspondência foi encontrada. Os valores na sua base são: {matches}")
else:
    print("\n❌ FALHA: Nenhuma correspondência exata encontrada.")
    print("Compare a string que buscamos acima com a lista de 'Valores únicos' para encontrar a diferença.")