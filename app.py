# === 1. IMPORTAÇÕES ===
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer

# === 0. CONFIG ===
st.set_page_config(page_title="Geração de Devolutivas", layout="wide")

# === 1. FUNÇÕES DE CACHE ===
@st.cache_resource
def carregar_modelo(nome_modelo: str, usar_cosseno: bool):
    if nome_modelo == "MiniLM (L2)":
        return SentenceTransformer("all-MiniLM-L6-v2")
    else:
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
    return pd.read_csv("data/Devolutivas.csv", sep=";")

@st.cache_data
def carregar_rubricas():
    return pd.read_csv("data/Rubricas.csv", sep=";")

# === 2. CARREGAMENTO ===
modelo_selecionado = st.sidebar.selectbox("Escolha o modelo de similaridade:", [
    "MiniLM (L2)",
    "Stella v1.5 (Cosseno)"
])

if modelo_selecionado == "MiniLM (L2)":
    df_odas = carregar_metadados("data/odas/metadados_odas.pkl")
    index = carregar_index("data/odas/odas_index.faiss")
    modelo = carregar_modelo("MiniLM (L2)", usar_cosseno=False)
    usar_cosseno = False
else:
    df_odas = carregar_metadados("data/odas/metadados_odas_stellav5.pkl")
    index = carregar_index("data/odas/odas_index_stellav5.faiss")
    modelo = carregar_modelo("Stella v1.5", usar_cosseno=True)
    usar_cosseno = True

# Carregar devolutivas e rubricas novos
df_devolutivas = carregar_devolutivas().rename(columns={
    "Necessidaes formativas": "Necessidades formativas"
})
df_rubricas = carregar_rubricas()

# === 3. TRATAMENTO DE DURAÇÃO ===
def limpar_descricao_antiga(texto):
    texto_limpo = re.sub(r"[📚🎥🧑‍🏫📘📄🎬⏱️]+", "", texto)
    texto_limpo = re.sub(r"\(.*?\)", "", texto)
    return texto_limpo.strip()

def interpretar_duracao(duracao):
    if pd.isna(duracao) or duracao.strip().lower() in ['s/d', '']:
        return "⏱️ Duração não informada"
    texto = limpar_descricao_antiga(str(duracao).lower())

    if any(p in texto for p in ["hora", "min", ":"]):
        numeros = [int(x) for x in re.findall(r"\d+", texto)]
        minutos = 0
        if len(numeros) == 1:
            minutos = numeros[0]
        elif len(numeros) == 2:
            minutos = numeros[0] * 60 + numeros[1]
        elif len(numeros) >= 3:
            minutos = numeros[0] * 60 + numeros[1] + numeros[2] // 60

        if minutos <= 5:
            return f"🎥 {texto} (vídeo curto)"
        elif minutos <= 20:
            return f"🎬 {texto} (vídeo médio)"
        else:
            return f"🧑‍🏫 {texto} (vídeo longo)"
    elif "página" in texto or texto.isdigit():
        paginas = int(re.findall(r"\d+", texto)[0])
        if paginas <= 3:
            return f"📄 {texto} (texto curto)"
        elif paginas <= 20:
            return f"📘 {texto} (texto médio)"
        else:
            return f"📚 {texto} (texto longo)"
    return f"⏱️ {texto}"

df_odas["Descricao_duracao"] = df_odas["Descricao_duracao"].apply(interpretar_duracao)

# === 4. DEVOLUTIVA (com Rubricas novas) ===

def encontrar_rubrica(pontuacao, dimensao, subdimensao):
    candidatos = df_rubricas[
        (df_rubricas["dimensao"] == dimensao) &
        (df_rubricas["subdimensao"] == subdimensao) &
        (df_rubricas["faixa_total_min"] <= pontuacao) &
        (df_rubricas["faixa_total_max"] >= pontuacao)
    ]
    if candidatos.empty:
        return None, None, None

    rubrica_numero = candidatos.iloc[0]["rubrica_numero"]
    rubrica_nome = candidatos.iloc[0]["rubrica_nome"]

    faixa_escolhida = candidatos[
        (candidatos["subfaixa_min"] <= pontuacao) &
        (candidatos["subfaixa_max"] >= pontuacao)
    ]
    if faixa_escolhida.empty:
        return rubrica_numero, rubrica_nome, None

    tipo_faixa = faixa_escolhida.iloc[0]["tipo_faixa"]
    return rubrica_numero, rubrica_nome, tipo_faixa

def gerar_devolutiva_markdown(pontuacao, dimensao, subdimensao):
    rubrica_numero, rubrica_nome, tipo_faixa = encontrar_rubrica(pontuacao, dimensao, subdimensao)

    if not rubrica_numero or not tipo_faixa:
        return "❌ Pontuação fora da faixa válida."

    rubrica_nome_completo = f"{rubrica_nome} – Nível {tipo_faixa}"

    devolutiva = df_devolutivas[
        (df_devolutivas["Dimensão"] == dimensao) &
        (df_devolutivas["Subdimensão"] == subdimensao) &
        (df_devolutivas["Rubrica numero"] == rubrica_numero) &
        (df_devolutivas["Rubrica nome"] == rubrica_nome_completo)
    ]

    if devolutiva.empty:
        return f"❌ Não foi encontrada devolutiva para Rubrica {rubrica_numero} - {rubrica_nome_completo}."

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

# === 5. EMBEDDING ===
def gerar_embedding_para_rag(texto):
    if "**Necessidades formativas:**" in texto:
        trecho = texto.split("**Necessidades formativas:**")[-1].strip()
    else:
        trecho = texto
    emb = modelo.encode([trecho])
    if usar_cosseno:
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb

# === 6. INTERFACE FINAL ===
st.title("📘 Geração de Devolutivas e Materiais Relacionados")

dimensao = st.selectbox("Escolha a dimensão:", sorted(df_devolutivas["Dimensão"].unique()))
subdimensoes = df_devolutivas[df_devolutivas["Dimensão"] == dimensao]["Subdimensão"].unique()
subdimensao = st.selectbox(
    "Escolha a subdimensão:",
    options=sorted(subdimensoes),
    index=0
)
pontuacao = st.slider("Pontuação:", 0, 51, 17)

if st.button("Gerar devolutiva"):
    texto_devolutiva = gerar_devolutiva_markdown(pontuacao, dimensao, subdimensao)
    if "❌" in texto_devolutiva:
        st.warning(texto_devolutiva)
    else:
        st.markdown(texto_devolutiva)

        embedding = gerar_embedding_para_rag(texto_devolutiva)
        top = 50
        distancias, indices = index.search(np.array(embedding).astype("float32"), top)
        resultados = df_odas.iloc[indices[0]].copy()
        resultados["distância"] = distancias[0]

        tipo_metric = "Cosseno" if usar_cosseno else "L2"
        st.markdown(f"### 📚 **Materiais recomendados com base na sua devolutiva (TOP {top}):**")
        for i, row in resultados.iterrows():
            titulo = row.get("Título", "Sem título")
            link = row.get("Fonte", "#")
            resumo = re.sub(r"<[^>]+>", "", str(row.get("Resumo", "Sem resumo disponível")).strip())
            suporte = row.get("Suporte", "Não informado")
            dim = row.get("Dimensões", "Não informado")
            duracao = row.get("Descricao_duracao", "⏱️ Duração não informada")
            similaridade = row["distância"]

            st.markdown(f"""
**{i+1}. [{titulo}]({link})**

📝 **Resumo:** {resumo}  
📎 **Tipo:** {suporte} | **Dimensão:** {dim}  
⏱️ **Duração:** {duracao}  
📏 **Similaridade ({tipo_metric}):** {similaridade:.4f}  

---
""")