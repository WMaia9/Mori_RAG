# utils.py

import streamlit as st
import pandas as pd
import numpy as np  # <--- ADICIONE ESTA LINHA
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer

# === FUNÇÕES DE CACHE PARA CARREGAMENTO DE DADOS ===

@st.cache_resource
def carregar_modelo_st():
    """Carrega o modelo de embedding."""
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

@st.cache_resource
def carregar_index(caminho: str):
    """Carrega o índice FAISS do disco."""
    return faiss.read_index(caminho)

@st.cache_data
def carregar_metadados(caminho: str):
    """Carrega o DataFrame de metadados do disco."""
    with open(caminho, "rb") as f:
        return pickle.load(f)

@st.cache_data
def carregar_devolutivas():
    """Carrega o CSV com os textos das devolutivas."""
    df = pd.read_csv("data/devolutivas.csv", sep=";")
    return df.rename(columns={"Necessidaes formativas": "Necessidades formativas"})

@st.cache_data
def carregar_rubricas():
    """Carrega o CSV com as faixas de pontuação das rubricas."""
    return pd.read_csv("data/rubricas.csv", sep=";")

# === FUNÇÕES AUXILIARES DE LÓGICA E FORMATAÇÃO ===

def encontrar_rubrica(df_rubricas, pontuacao, dimensao, subdimensao):
    candidatos = df_rubricas[
        (df_rubricas['dimensao'] == dimensao) &
        (df_rubricas['subdimensao'] == subdimensao) &
        (df_rubricas['faixa_total_min'] <= pontuacao) &
        (df_rubricas['faixa_total_max'] >= pontuacao)
    ]
    if candidatos.empty: return None, None, None
    rubrica_numero = candidatos.iloc[0]['rubrica_numero']
    rubrica_nome = candidatos.iloc[0]['rubrica_nome']
    faixa = candidatos[
        (candidatos['subfaixa_min'] <= pontuacao) &
        (candidatos['subfaixa_max'] >= pontuacao)
    ]
    if faixa.empty: return rubrica_numero, rubrica_nome, None
    tipo_faixa = faixa.iloc[0]['tipo_faixa']
    return rubrica_numero, rubrica_nome, tipo_faixa

def formatar_necessidades_formativas(texto):
    if texto is None or not isinstance(texto, str) or texto.strip() == "" or pd.isna(texto):
        return "Sem necessidades formativas informadas."
    linhas = texto.strip().split("\n")
    markdown_final = ""
    for linha in linhas:
        if not linha.strip(): continue
        partes = [p.strip() for p in linha.split("•") if p.strip()]
        if not partes: continue
        markdown_final += f"\n- **{partes[0]}**\n"
        for detalhe in partes[1:]:
            markdown_final += f"  - {detalhe}\n"
    return markdown_final.strip()

# VERSÃO NOVA E CORRIGIDA

def gerar_texto_devolutiva_markdown(df_devolutivas, df_rubricas, pontuacao, dimensao, subdimensao):
    """
    Gera o texto completo da devolutiva com a formatação final corrigida,
    usando negrito em vez de cabeçalhos para os subtítulos.
    """
    rubrica_numero, rubrica_nome, tipo_faixa = encontrar_rubrica(df_rubricas, pontuacao, dimensao, subdimensao)
    if rubrica_numero is None or tipo_faixa is None:
        st.warning(f"Não foi encontrada uma rubrica ou faixa de nível correspondente para a pontuação {pontuacao} na subdimensão '{subdimensao}'. Verifique o arquivo de rubricas.")
        return None

    devolutiva = df_devolutivas[
        (df_devolutivas['Dimensão'] == dimensao) &
        (df_devolutivas['Subdimensão'] == subdimensao) &
        (df_devolutivas['Rubrica numero'] == rubrica_numero) &
        (df_devolutivas['Rubrica nome'] == f"{rubrica_nome} – Nível {tipo_faixa}")
    ]
    if devolutiva.empty:
        st.warning(f"O texto da devolutiva não foi encontrado para a Rubrica {rubrica_numero} - Nível {tipo_faixa}. Verifique o arquivo de devolutivas.")
        return None
        
    item = devolutiva.iloc[0]
    
    # AQUI ESTÁ A CORREÇÃO: trocamos '###' por '**' e garantimos a linha em branco.
    return f"""
## 📄 Devolutiva Personalizada

- 🔢 **Pontuação:** {pontuacao}
- 📂 **Dimensão:** {dimensao}
- 📁 **Subdimensão:** {subdimensao}
- 🏷️ **Rubrica:** Rubrica {rubrica_numero} - {rubrica_nome}
- 📊 **Nível:** {tipo_faixa}

---

**✅ Seus pontos fortes:**

{item['Pontos fortes']}

---

**📈 O que fazer para avançar:**

{item['O que fazer para avançar']}

---

**📚 Necessidades formativas:**

{formatar_necessidades_formativas(item['Necessidades formativas'])}
""".strip()

def gerar_texto_devolutiva_rico(df_devolutivas, df_rubricas, pontuacao, dimensao, subdimensao, modelo_selecionado):
    rubrica_numero, rubrica_nome, tipo_faixa = encontrar_rubrica(df_rubricas, pontuacao, dimensao, subdimensao)
    if rubrica_numero is None or tipo_faixa is None: return None
    devolutiva = df_devolutivas[
        (df_devolutivas['Dimensão'] == dimensao) &
        (df_devolutivas['Subdimensão'] == subdimensao) &
        (df_devolutivas['Rubrica numero'] == rubrica_numero) &
        (df_devolutivas['Rubrica nome'] == f"{rubrica_nome} – Nível {tipo_faixa}")
    ]
    if devolutiva.empty: return None
    item = devolutiva.iloc[0]
    if modelo_selecionado == "Modelo Avançado (v2, Re-ranking)":
        contexto_query = f"Perfil do usuário: gestor no Nível {tipo_faixa} da Rubrica {rubrica_numero} - {rubrica_nome}. A necessidade de aprendizagem é a seguinte:"
        return f"{contexto_query}\n\n{item['Necessidades formativas']}".strip()
    else:
        return f"Necessidades formativas:\n{item['Necessidades formativas']}".strip()

def gerar_embedding_para_rag(modelo_st, texto: str) -> np.ndarray:
    embedding = modelo_st.encode([texto])
    return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

def gerar_card_material(row, i):
    titulo = row.get("Título", "Sem título")
    resumo = re.sub(r"<[^>]+>", "", str(row.get("Resumo", "Sem resumo disponível")).strip())
    suporte = row.get("Suporte", "Não informado")
    tipo = row.get("Tipo", "Não informado")
    dimensao_card = row.get("Dimensões", "Não informado")
    duracao = row.get("Descricao_duracao", "⏱️ Duração não informada")
    link_real = str(row.get("Fonte", "#")).strip()
    if link_real.lower() == "nan" or link_real == "": link_real = "#"
    sim = float(row.get('distância', 0.0))
    interpretacao = ""
    if sim > 0:
        if sim >= 0.80: interpretacao = "🔥 Altamente relevante"
        elif sim >= 0.65: interpretacao = "✅ Relevante"
        elif sim >= 0.50: interpretacao = "🧐 Moderadamente relevante"
        else: interpretacao = "🔍 Pouco relevante"
        interpretacao = f"– *{interpretacao}*"
    return f"""
**{i+1}. [{titulo}]({link_real})**
- 📝 **Resumo:** {resumo}
- 📎 **Tipo:** {suporte} | **Subtipo:** {tipo}
- 📂 **Dimensão:** {dimensao_card}
- ⏱️ **Duração:** {duracao}
- 📏 **Similaridade Ponderada:** {sim:.4f} {interpretacao}
---
"""

def obter_pontuacao_maxima(df_rubricas, dimensao, subdimensao):
    rubricas_filtradas = df_rubricas[
        (df_rubricas['dimensao'] == dimensao) &
        (df_rubricas['subdimensao'] == subdimensao)
    ]
    if rubricas_filtradas.empty: return 51
    return int(rubricas_filtradas['faixa_total_max'].max())