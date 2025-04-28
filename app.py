# === 1. IMPORTAÇÕES ===
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer

# === 2. CONFIGURAÇÃO INICIAL ===
st.set_page_config(page_title="📘 Geração de Devolutivas e Materiais", layout="wide")

# === 3. FUNÇÕES DE CACHE ===
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

@st.cache_resource
def carregar_index(caminho: str):
    return faiss.read_index(caminho)

@st.cache_data
def carregar_metadados(caminho: str):
    with open(caminho, "rb") as f:
        return pickle.load(f)

@st.cache_data
def carregar_devolutivas():
    df = pd.read_csv("data/Devolutivas.csv", sep=";")
    return df.rename(columns={"Necessidaes formativas": "Necessidades formativas"})

@st.cache_data
def carregar_rubricas():
    return pd.read_csv("data/Rubricas.csv", sep=";")

# === 4. CARREGAMENTO DOS DADOS ===
modelo = carregar_modelo()
index = carregar_index("data/odas/odas_index_stellav5.faiss")
df_odas = carregar_metadados("data/odas/metadados_odas_stellav5.pkl")
df_devolutivas = carregar_devolutivas()
df_rubricas = carregar_rubricas()

# === 5. FUNÇÕES DE APOIO ===
def encontrar_rubrica(pontuacao, dimensao, subdimensao):
    candidatos = df_rubricas[
        (df_rubricas['dimensao'] == dimensao) &
        (df_rubricas['subdimensao'] == subdimensao) &
        (df_rubricas['faixa_total_min'] <= pontuacao) &
        (df_rubricas['faixa_total_max'] >= pontuacao)
    ]
    if candidatos.empty:
        return None, None, None

    rubrica_numero = candidatos.iloc[0]['rubrica_numero']
    rubrica_nome = candidatos.iloc[0]['rubrica_nome']

    faixa = candidatos[
        (candidatos['subfaixa_min'] <= pontuacao) &
        (candidatos['subfaixa_max'] >= pontuacao)
    ]
    if faixa.empty:
        return rubrica_numero, rubrica_nome, None

    tipo_faixa = faixa.iloc[0]['tipo_faixa']
    return rubrica_numero, rubrica_nome, tipo_faixa

def gerar_texto_devolutiva_markdown(pontuacao, dimensao, subdimensao):
    rubrica_numero, rubrica_nome, tipo_faixa = encontrar_rubrica(pontuacao, dimensao, subdimensao)
    if not rubrica_numero or not tipo_faixa:
        return None

    devolutiva = df_devolutivas[
        (df_devolutivas['Dimensão'] == dimensao) &
        (df_devolutivas['Subdimensão'] == subdimensao) &
        (df_devolutivas['Rubrica numero'] == rubrica_numero) &
        (df_devolutivas['Rubrica nome'] == f"{rubrica_nome} – Nível {tipo_faixa}")
    ]

    if devolutiva.empty:
        return None

    item = devolutiva.iloc[0]

    return f"""
## 📄 **Devolutiva personalizada:**

🔢 **Pontuação:** {pontuacao}  
📂 **Dimensão:** {dimensao}  
📁 **Subdimensão:** {subdimensao}  
🏷️ **Rubrica:** Rubrica {rubrica_numero} - {rubrica_nome}  
📊 **Nível:** {tipo_faixa}

---

✅ **Seus pontos fortes:**  
{item['Pontos fortes']}

---

📈 **O que fazer para avançar:**  
{item['O que fazer para avançar']}

---

📚 **Necessidades formativas:**  
{item['Necessidades formativas']}
""".strip()

def gerar_texto_devolutiva_rico(pontuacao, dimensao, subdimensao):
    # Versão usada para gerar embeddings (sem formatação)
    rubrica_numero, rubrica_nome, tipo_faixa = encontrar_rubrica(pontuacao, dimensao, subdimensao)
    if not rubrica_numero or not tipo_faixa:
        return None

    devolutiva = df_devolutivas[
        (df_devolutivas['Dimensão'] == dimensao) &
        (df_devolutivas['Subdimensão'] == subdimensao) &
        (df_devolutivas['Rubrica numero'] == rubrica_numero) &
        (df_devolutivas['Rubrica nome'] == f"{rubrica_nome} – Nível {tipo_faixa}")
    ]

    if devolutiva.empty:
        return None

    item = devolutiva.iloc[0]

    return f"""
Dimensão: {dimensao}
Subdimensão: {subdimensao}
Rubrica: Rubrica {rubrica_numero} - {rubrica_nome}
Nível: {tipo_faixa}

Seus pontos fortes:
{item['Pontos fortes']}

O que fazer para avançar:
{item['O que fazer para avançar']}

Necessidades formativas:
{item['Necessidades formativas']}
""".strip()

def gerar_embedding_para_rag(texto: str) -> np.ndarray:
    embedding = modelo.encode([texto])
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding

def interpretar_similaridade(valor):
    if valor >= 0.80:
        return "🔥 Altamente relevante"
    elif valor >= 0.65:
        return "✅ Relevante"
    elif valor >= 0.50:
        return "🧐 Moderadamente relevante"
    else:
        return "🔍 Pouco relevante"

def gerar_card_material(row, i):
    titulo = row.get("Título", "Sem título")
    resumo = re.sub(r"<[^>]+>", "", str(row.get("Resumo", "Sem resumo disponível")).strip())
    suporte = row.get("Suporte", "Não informado")
    dimensao = row.get("Dimensões", "Não informado")
    duracao = row.get("Descricao_duracao", "⏱️ Duração não informada")
    link_real = str(row.get("Fonte", "#")).strip()
    if link_real.lower() == "nan" or link_real == "":
        link_real = "#"
    sim = float(row['distância'])
    interpretacao = interpretar_similaridade(sim)

    return f"""
**{i+1}. [{titulo}]({link_real})**
- 📝 **Resumo:** {resumo}
- 📎 **Tipo:** {suporte} | **Dimensão:** {dimensao}
- ⏱️ **Duração:** {duracao}
- 📏 **Similaridade:** {sim:.4f} – *{interpretacao}*

---
"""

def obter_pontuacao_maxima(dimensao, subdimensao):
    rubricas_filtradas = df_rubricas[
        (df_rubricas['dimensao'] == dimensao) &
        (df_rubricas['subdimensao'] == subdimensao)
    ]
    if rubricas_filtradas.empty:
        return 51  # fallback se não encontrar
    return int(rubricas_filtradas['faixa_total_max'].max())


# === 6. INTERFACE ===
st.title("📘 Geração de Devolutivas e Materiais Relacionados")

dimensao = st.selectbox("Escolha a dimensão:", sorted(df_devolutivas["Dimensão"].unique()))
subdimensoes = df_devolutivas[df_devolutivas["Dimensão"] == dimensao]["Subdimensão"].unique()
subdimensao = st.selectbox("Escolha a subdimensão:", sorted(subdimensoes))

pontuacao_max = obter_pontuacao_maxima(dimensao, subdimensao)
pontuacao = st.slider("Pontuação:", 0, pontuacao_max, min(17, pontuacao_max))

if st.button("Gerar devolutiva"):
    texto_markdown = gerar_texto_devolutiva_markdown(pontuacao, dimensao, subdimensao)
    texto_rico = gerar_texto_devolutiva_rico(pontuacao, dimensao, subdimensao)

    if texto_markdown is None or texto_rico is None:
        st.warning("⚠️ Não foi possível gerar devolutiva para essa pontuação.")
    else:
        st.markdown(texto_markdown)

        embedding = gerar_embedding_para_rag(texto_rico)
        distancias, indices = index.search(np.array(embedding).astype("float32"), 50)
        resultados = df_odas.iloc[indices[0]].copy()
        resultados["distância"] = distancias[0]

        # Apenas em português
        resultados = resultados[resultados["Idiomas"].str.contains("português", case=False, na=False)]

        artigos = resultados[resultados["Suporte"].str.contains("Texto|Artigo|Livro", case=False, na=False)]
        videos = resultados[resultados["Suporte"].str.contains("Vídeo|Curso|Aula", case=False, na=False)]

        markdown = "## 📚 **Materiais recomendados – Artigos:**\n\n"
        for i, row in artigos.iterrows():
            markdown += gerar_card_material(row, i)

        markdown += "\n\n## 🎥 **Materiais recomendados – Vídeos:**\n\n"
        for i, row in videos.iterrows():
            markdown += gerar_card_material(row, i)

        st.markdown(markdown)