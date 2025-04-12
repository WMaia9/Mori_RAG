import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# === Configuração inicial ===
st.set_page_config(page_title="Geração de Devolutivas", layout="wide")
st.title("📘 Geração de Devolutivas e Materiais Relacionados")

# === Escolha do modelo ===
modelo_selecionado = st.radio(
    "Escolha o modelo de embeddings:",
    options=["MiniLM (L2)", "Stella v1.5 (Cosseno)"],
    index=0
)

# Escolha de arquivos com base no modelo
if modelo_selecionado == "MiniLM (L2)":
    with open("data/odas/metadados_odas.pkl", "rb") as f:
        df_odas = pickle.load(f)
    index = faiss.read_index("data/odas/odas_index.faiss")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    usar_cosseno = False
else:
    with open("data/odas/metadados_odas_stellav5.pkl", "rb") as f:
        df_odas = pickle.load(f)
    index = faiss.read_index("data/odas/odas_index_stellav5.faiss")
    modelo = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    usar_cosseno = True

# === Carregar devolutivas ===
df_devolutivas = pd.read_csv("data/devolutivas_padronizadas.csv", sep=";")

# Corrigir coluna de duração
def classificar_duracao(valor):
    if pd.isna(valor):
        return "⏱️ Duração não informada"
    texto = str(valor)
    texto_limpo = re.sub(r"[📚🎥🧑‍🏫📘📄🎬🎞️⏱️]+", "", texto).strip()
    return f"{texto_limpo}"

df_odas["Descricao_duracao"] = df_odas["Descricao_duracao"].apply(classificar_duracao)

# === Funções auxiliares ===
def pontuacao_para_rubrica_nivel(pontuacao):
    if 0 <= pontuacao <= 4:
        return "Rubrica 1 - Sensibilização", "Consolidar"
    elif 5 <= pontuacao <= 9:
        return "Rubrica 1 - Sensibilização", "Avançar"
    elif 10 <= pontuacao <= 16:
        return "Rubrica 2 - Exploração", "Consolidar"
    elif 17 <= pontuacao <= 21:
        return "Rubrica 2 - Exploração", "Avançar"
    elif 22 <= pontuacao <= 29:
        return "Rubrica 3 - Liderança estratégica", "Consolidar"
    elif 30 <= pontuacao <= 36:
        return "Rubrica 3 - Liderança estratégica", "Avançar"
    elif 37 <= pontuacao <= 45:
        return "Rubrica 4 - Transformação cultural", "Consolidar"
    return None, None

def gerar_devolutiva(pontuacao, dimensao, subdimensao):
    rubrica, nivel = pontuacao_para_rubrica_nivel(pontuacao)
    if not rubrica:
        return "❌ Pontuação fora da faixa válida.", ""
    resultado = df_devolutivas[
        (df_devolutivas["dimensao"] == dimensao)
        & (df_devolutivas["subdimensao"] == subdimensao)
        & (df_devolutivas["rubrica"] == rubrica)
        & (df_devolutivas["nivel"] == nivel)
    ]
    if resultado.empty:
        return f"❌ Não foi encontrada a devolutiva para {rubrica} - {nivel}.", ""
    item = resultado.iloc[0]
    texto = f"""
🔢 **Pontuação:** {pontuacao}  
📂 **Dimensão:** {dimensao}  
📁 **Subdimensão:** {subdimensao}  
🏷️ **Rubrica:** {rubrica}  
📊 **Nível:** {nivel}  

---

✅ **Seus pontos fortes:**  
{item['pontos_fortes']}

---

📈 **O que fazer para avançar:**  
{item['avancar']}

---

📚 **Necessidades formativas:**  
{item['formativas']}
"""
    return "", texto.strip()

def gerar_embedding_para_rag(texto):
    if "**Necessidades formativas:**" in texto:
        trecho = texto.split("**Necessidades formativas:**")[-1].strip()
    else:
        trecho = texto
    embedding = modelo.encode([trecho])
    if usar_cosseno:
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding

# === Interface principal ===
dimensao = st.selectbox("Escolha a dimensão:", sorted(df_devolutivas["dimensao"].unique()))
subdimensoes = df_devolutivas[df_devolutivas["dimensao"] == dimensao]["subdimensao"].unique()
subdimensao = st.selectbox(
    "Escolha a subdimensão:",
    options=sorted(subdimensoes),
    index=sorted(subdimensoes).index("planejamento") if "planejamento" in subdimensoes else 0
)
pontuacao = st.slider("Pontuação:", 0, 45, 17)

if st.button("Gerar devolutiva"):
    erro, texto_devolutiva = gerar_devolutiva(pontuacao, dimensao, subdimensao)
    if erro:
        st.warning(erro)
    else:
        st.markdown("### 📄 **Devolutiva personalizada:**")
        st.markdown(texto_devolutiva)

        embedding = gerar_embedding_para_rag(texto_devolutiva)
        top = 50
        distancias, indices = index.search(np.array(embedding).astype("float32"), top)
        resultados = df_odas.iloc[indices[0]].copy()
        resultados["distância"] = distancias[0]

        st.markdown(f"### 📚 **Materiais recomendados com base na sua devolutiva (TOP {top}):**")
        for i, row in resultados.iterrows():
            titulo = row.get("Título", "Sem título")
            link = row.get("Fonte", "#")
            resumo_raw = str(row.get("Resumo", "Sem resumo disponível"))
            resumo = re.sub(r"<[^>]+>", "", resumo_raw).strip()
            suporte = row.get("Suporte", "Não informado")
            dim = row.get("Dimensões", "Não informado")
            duracao = row.get("Descricao_duracao", "⏱️ Duração não informada")
            similaridade = row["distância"]

            if usar_cosseno:
                metrica = f"📏 **Similaridade (Cosseno):** {similaridade:.4f}"
            else:
                metrica = f"📏 **Distância L2:** {similaridade:.4f}"

            st.markdown(f"""
**{i+1}. [{titulo}]({link})**

📝 **Resumo:** {resumo}  
📎 **Tipo:** {suporte} | **Dimensão:** {dim}  
⏱️ **Duração:** {duracao}  
{metrica}

---
""")