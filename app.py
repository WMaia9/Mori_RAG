import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer

# === 0. CONFIG ===
st.set_page_config(page_title="Geração de Devolutivas", layout="wide")

# === 1. MODELO ===
modelo_selecionado = st.sidebar.selectbox("Escolha o modelo de similaridade:", [
    "MiniLM (L2)",
    "Stella v1.5 (Cosseno)"
])

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

# === 2. Limpeza e categorização da duração ===
def limpar_descricao_antiga(texto):
    texto_limpo = re.sub(r"[📚🎥🧑‍🏫📘📄🎬⏱️]+", "", texto)
    texto_limpo = re.sub(r"\(.*?\)", "", texto_limpo)
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

# === 3. Carregar devolutivas ===
df_devolutivas = pd.read_csv("data/devolutivas_padronizadas.csv", sep=";")

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

# === 4. Embedding e busca ===
def gerar_embedding_para_rag(texto):
    if "**Necessidades formativas:**" in texto:
        trecho = texto.split("**Necessidades formativas:**")[-1].strip()
    else:
        trecho = texto
    emb = modelo.encode([trecho])
    if usar_cosseno:
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb

# === 5. Interface ===
st.title("📘 Geração de Devolutivas e Materiais Relacionados")

dimensao = st.selectbox("Escolha a dimensão:", sorted(df_devolutivas["dimensao"].unique()))
subdimensoes = df_devolutivas[df_devolutivas["dimensao"] == dimensao]["subdimensao"].unique()
subdimensao = st.selectbox(
    "Escolha a subdimensão:",
    options=sorted(subdimensoes),
    index=sorted(subdimensoes).index("planejamento") if "planejamento" in subdimensoes else 0
)
pontuacao = st.slider("Pontuação:", 0, 45, 17)

# Botão de execução
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

        tipo_metric = "Cosseno" if usar_cosseno else "L2"
        st.markdown(f"### 📚 **Materiais recomendados com base na sua devolutiva TOP {top}:**")
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