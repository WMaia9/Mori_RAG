# === 1. IMPORTAÇÕES ===
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI

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
    df = pd.read_csv("data/devolutivas.csv", sep=";")
    return df.rename(columns={"Necessidaes formativas": "Necessidades formativas"})

@st.cache_data
def carregar_rubricas():
    return pd.read_csv("data/rubricas.csv", sep=";")

# === 4. CARREGAMENTO DOS DADOS ===
modelo = carregar_modelo()
df_devolutivas = carregar_devolutivas()
df_rubricas = carregar_rubricas()

# === 4.1. CONFIGURAÇÕES NA BARRA LATERAL (SIDEBAR) ===
st.sidebar.markdown("### 🔍 Configurações de Recomendação")
modelo_ativo = st.sidebar.selectbox(
    "Escolha o modelo de busca:",
    ["New model", "Old model"],
    help="O modelo de busca define a base de dados de materiais que será consultada."
)

st.sidebar.markdown("### 🤖 Configurações de IA (GPT)")
modelo_gpt_selecionado = st.sidebar.selectbox(
    "Escolha o modelo de IA:",
    ["gpt-4o-mini", "gpt-4"],
    index=0,  # gpt-4o-mini será o padrão
    help="gpt-4o-mini é mais rápido e econômico. gpt-4 é mais poderoso, mas mais lento e caro."
)

# === 4.2. CARREGAMENTO CONDICIONAL DOS DADOS DE BUSCA ===
if modelo_ativo == "Old model":
    index = carregar_index("data/odas/odas_index_stellav5.faiss")
    df_odas = carregar_metadados("data/odas/metadados_odas_stellav5.pkl")
else: # New model
    index = carregar_index("data/odas/odas_index_vnova.faiss")
    df_odas = carregar_metadados("data/odas/metadados_odas_vnova.pkl")


# === 5. FUNÇÕES AUXILIARES ===
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

def formatar_necessidades_formativas(texto):
    if texto is None or not isinstance(texto, str) or texto.strip() == "" or pd.isna(texto):
        return "Sem necessidades formativas informadas."

    linhas = texto.strip().split("\n")
    markdown_final = ""

    for linha in linhas:
        if not linha.strip():
            continue
        partes = [p.strip() for p in linha.split("•") if p.strip()]
        if len(partes) == 0:
            continue
        if len(partes) == 1:
            markdown_final += f"\n- **{partes[0]}**\n"
        else:
            markdown_final += f"\n- **{partes[0]}**\n"
            for detalhe in partes[1:]:
                markdown_final += f"  - {detalhe}\n"
    return markdown_final.strip()

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
{formatar_necessidades_formativas(item['Necessidades formativas'])}
""".strip()

def gerar_texto_devolutiva_rico(pontuacao, dimensao, subdimensao):
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
    tipo = row.get("Tipo", "Não informado")
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
- 📎 **Tipo:** {suporte} | **Subtipo:** {tipo}
- 📂 **Dimensão:** {dimensao}
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
        return 51
    return int(rubricas_filtradas['faixa_total_max'].max())

def reordenar_materiais_com_gpt(client, materiais_df, tipo_material, texto_rico, pontuacao, pontuacao_max, subdimensao, nivel_usuario, modelo_gpt: str):
    if materiais_df.empty:
        return materiais_df

    prompt = f"""
Você é um especialista em formação docente. Reordene os materiais abaixo de acordo com o seguinte perfil:

- Pontuação do usuário: {pontuacao}/{pontuacao_max} em "{subdimensao}"
- Perfil: nível {nivel_usuario}
- Devolutiva textual:
\"\"\"{texto_rico}\"\"\"

Abaixo estão {len(materiais_df)} {tipo_material}. Para cada um:
- Analise o título e o resumo para entender o conteúdo.
- Crie uma ordem de recomendação ideal (o mais útil primeiro) para o perfil do usuário.
- Priorize materiais com durações menores, pois o usuário pode ter pouco tempo.
- Atribua uma nota de 1 a 5 estrelas (⭐) com base na relevância para o usuário.
- Justifique sua recomendação para cada item em 1 parágrafo curto.

Formato da resposta (use exatamente este formato):
1. Título do material – ⭐⭐⭐⭐
Justificativa: ...
2. Título do outro material – ⭐⭐⭐
Justificativa: ...
"""
    for index, row in materiais_df.iterrows():
        titulo = row.get("Título", "Sem título")
        resumo = re.sub(r"<[^>]+>", "", str(row.get("Resumo", "Sem resumo")).strip())
        prompt += f"\n\n- Título: {titulo}\n  Resumo: {resumo}"

    try:
        response = client.chat.completions.create(
            model=modelo_gpt,
            messages=[
                {"role": "system", "content": "Você é um especialista em formação de professores e gestores escolares."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2500
        )
        resposta_gpt = response.choices[0].message.content

        ordem_titulos = []
        for linha in resposta_gpt.strip().split("\n"):
            match = re.match(r"^\d+\.\s*(.+?)\s*–\s*[⭐★]{1,5}", linha.strip())
            if match:
                titulo = match.group(1).strip().lower()
                ordem_titulos.append(titulo)

        if not ordem_titulos:
            st.warning(f"Não foi possível extrair a ordem para {tipo_material} da resposta do GPT. Exibindo na ordem de similaridade.")
            return materiais_df

        materiais_df["título_clean"] = materiais_df["Título"].str.lower().str.strip()
        materiais_reordenados = []

        for titulo_gpt in ordem_titulos:
            match = materiais_df[materiais_df["título_clean"].str.contains(titulo_gpt, case=False, na=False, regex=False)]
            if not match.empty:
                item_encontrado = match.iloc[0]
                materiais_reordenados.append(item_encontrado)
                materiais_df = materiais_df.drop(item_encontrado.name)

        if not materiais_df.empty:
            materiais_reordenados.extend([row for _, row in materiais_df.iterrows()])

        return pd.DataFrame(materiais_reordenados)

    except Exception as e:
        st.error(f"Erro ao reordenar {tipo_material} com a IA: {str(e)}")
        return materiais_df

# === 6. INTERFACE ===
st.title("📘 Geração de Devolutivas e Materiais Relacionados")
modo = st.radio("Escolha o modo:", ["Individual", "Geral"], key="modo_selecao")

if modo == "Individual":
    dimensao = st.selectbox("Escolha a dimensão:", sorted(df_devolutivas["Dimensão"].unique()))
    subdimensoes = df_devolutivas[df_devolutivas["Dimensão"] == dimensao]["Subdimensão"].unique()
    subdimensao = st.selectbox("Escolha a subdimensão:", sorted(subdimensoes))

    pontuacao_max = obter_pontuacao_maxima(dimensao, subdimensao)
    pontuacao = st.slider("Pontuação:", 0, pontuacao_max, min(17, pontuacao_max))

    usar_gpt = st.toggle(
        "✨ Reordenar recomendações com IA (GPT)",
        value=True,
        help="Se ativado, usa o GPT para reordenar os materiais de acordo com seu perfil. Requer uma chave de API da OpenAI."
    )

    openai_api_key = ""
    if usar_gpt:
        openai_api_key = st.text_input("Insira sua OpenAI API Key para a reordenação", type="password")

    if st.button("Gerar devolutiva e recomendações"):
        texto_markdown = gerar_texto_devolutiva_markdown(pontuacao, dimensao, subdimensao)
        texto_rico = gerar_texto_devolutiva_rico(pontuacao, dimensao, subdimensao)

        if texto_markdown is None or texto_rico is None:
            st.warning("⚠️ Não foi possível gerar devolutiva para essa pontuação.")
        else:
            st.markdown(texto_markdown)
            
            nivel_percentual = pontuacao / pontuacao_max if pontuacao_max > 0 else 0
            if nivel_percentual < 0.20: nivel_usuario = "muito iniciante"
            elif nivel_percentual < 0.40: nivel_usuario = "iniciante"
            elif nivel_percentual < 0.60: nivel_usuario = "intermediário"
            elif nivel_percentual < 0.80: nivel_usuario = "avançado"
            else: nivel_usuario = "muito avançado"

            with st.spinner("Buscando materiais relevantes..."):
                embedding = gerar_embedding_para_rag(texto_rico)
                distancias, indices = index.search(np.array(embedding).astype("float32"), 250)
                resultados = df_odas.iloc[indices[0]].copy()
                resultados["distância"] = distancias[0]
                resultados = resultados[resultados["Idiomas"].str.contains("português", case=False, na=False)]

                artigos = resultados[resultados["Suporte"].str.contains("Texto|Artigo|Livro|Relatório|Resenha|Plano de aula", case=False, na=False)].head(15)
                videos = resultados[resultados["Suporte"].str.contains("Vídeo|Curso|Aula", case=False, na=False)].head(15)
                audios = resultados[resultados["Suporte"].str.contains("Áudio|Podcast|Rádio", case=False, na=False)].head(15)

            titulo_secao = "organizados por similaridade"
            if usar_gpt:
                if not openai_api_key:
                    st.error("Para usar a reordenação com IA, por favor, insira sua OpenAI API Key acima e clique no botão novamente.")
                    st.stop()

                st.info(f"💡 Reordenando materiais com o modelo {modelo_gpt_selecionado} para máxima relevância...")
                with st.spinner("Aguarde, a IA está personalizando suas recomendações..."):
                    client = OpenAI(api_key=openai_api_key)
                    artigos = reordenar_materiais_com_gpt(client, artigos, "materiais de texto", texto_rico, pontuacao, pontuacao_max, subdimensao, nivel_usuario, modelo_gpt=modelo_gpt_selecionado)
                    videos = reordenar_materiais_com_gpt(client, videos, "vídeos", texto_rico, pontuacao, pontuacao_max, subdimensao, nivel_usuario, modelo_gpt=modelo_gpt_selecionado)
                    audios = reordenar_materiais_com_gpt(client, audios, "áudios", texto_rico, pontuacao, pontuacao_max, subdimensao, nivel_usuario, modelo_gpt=modelo_gpt_selecionado)
                titulo_secao = "organizados por relevância para você"

            if not artigos.empty:
                st.markdown("---")
                st.markdown(f"### 📚 **Textos recomendados ({titulo_secao})**")
                for i, row in enumerate(artigos.itertuples()):
                    st.markdown(gerar_card_material(row._asdict(), i))

            if not videos.empty:
                st.markdown("---")
                st.markdown(f"### 🎥 **Vídeos recomendados ({titulo_secao})**")
                for i, row in enumerate(videos.itertuples()):
                    st.markdown(gerar_card_material(row._asdict(), i))

            if not audios.empty:
                st.markdown("---")
                st.markdown(f"### 🎧 **Áudios recomendados ({titulo_secao})**")
                for i, row in enumerate(audios.itertuples()):
                    st.markdown(gerar_card_material(row._asdict(), i))

elif modo == "Geral":
    st.markdown("### Escolha a dimensão que deseja gerar a devolutiva geral:")
    dimensao_escolhida = st.selectbox("Dimensão:", ["Planejamento pedagógico", "Pessoal-relacional"])

    if dimensao_escolhida == "Planejamento pedagógico":
        st.markdown("#### Informe as pontuações das subdimensões pedagógicas:")
        subdimensoes = [
            "Desenvolvimento profissional docente",
            "Implementação do processo de ensino e aprendizagem",
            "Monitoramento e Avaliação da Aprendizagem",
            "Planejamento pedagógico",
            "Proteção das Trajetórias Estudantis"
        ]
        pontuacoes = {}
        for sub in subdimensoes:
            max_ponto = obter_pontuacao_maxima("Dimensão pedagógica", sub)
            pontuacoes[sub] = st.slider(f"{sub}", 0, max_ponto, 0, key=f"slider_{sub}")

        openai_api_key = st.text_input("Insira sua OpenAI API Key", type="password", key="geral_api_key_1")

        if st.button("Gerar devolutiva da dimensão pedagógica") and openai_api_key:
            partes = []
            for sub, ponto in pontuacoes.items():
                texto = gerar_texto_devolutiva_rico(ponto, "Dimensão pedagógica", sub)
                if texto:
                    partes.append(f"Subdimensão {sub}:\n{texto}")

            if not partes:
                st.warning("⚠️ Nenhuma pontuação informada.")
            else:
                prompt = f"""
Você é um assistente especializado em gestão escolar. Seu objetivo é receber as devolutivas textuais de cada subdimensão e gerar um texto síntese único para a dimensão “Planejamento pedagógico”.
Tarefa:
- Identificar e sintetizar os principais pontos fortes que emergem de todas as subdimensões.
- Apontar as ações concretas que o gestor deve implementar para avançar ao próximo nível de maturidade (conforme rubricas).
- Limite: até 3 parágrafos.
- Tom: claro, direto, orientado a “próximos passos”.
{chr(10).join(partes)}
"""
                try:
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model=modelo_gpt_selecionado,
                        messages=[
                            {"role": "system", "content": "Você é um especialista em formação de professores."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    resposta = response.choices[0].message.content
                    st.markdown("### 📖 Devolutiva da Dimensão Pedagógica")
                    st.markdown(resposta)
                except Exception as e:
                    st.error(f"Erro ao gerar devolutiva: {str(e)}")

    elif dimensao_escolhida == "Pessoal-relacional":
        st.markdown("#### Informe a pontuação da subdimensão:")
        sub = "Convivência no ambiente escolar"
        max_ponto = obter_pontuacao_maxima("Dimensão pessoal-relacional", sub)
        ponto = st.slider(f"{sub}", 0, max_ponto, 0, key=f"slider_{sub}")

        openai_api_key = st.text_input("Insira sua OpenAI API Key", type="password", key="geral_api_key_2")

        if st.button("Gerar devolutiva da dimensão pessoal-relacional") and openai_api_key:
            texto = gerar_texto_devolutiva_rico(ponto, "Dimensão pessoal-relacional", sub)

            if not texto:
                st.warning("⚠️ Nenhuma pontuação informada.")
            else:
                prompt = f"""
Você é um assistente especializado em gestão escolar. Seu objetivo é receber as devolutivas textuais de cada subdimensão e gerar um texto síntese único para a dimensão “Pessoal-Relacional”.
Tarefa:
- Identificar e sintetizar os principais pontos fortes que emergem da subdimensão.
- Apontar as ações concretas que o gestor deve implementar para avançar ao próximo nível de maturidade (conforme rubricas).
- Limite: até 3 parágrafos.
- Tom: claro, direto, orientado a “próximos passos”.
Subdimensão {sub}:\n{texto}
"""
                try:
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model=modelo_gpt_selecionado,
                        messages=[
                            {"role": "system", "content": "Você é um especialista em formação de professores."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    resposta = response.choices[0].message.content
                    st.markdown("### 📖 Devolutiva da Dimensão Pessoal-Relacional")
                    st.markdown(resposta)
                except Exception as e:
                    st.error(f"Erro ao gerar devolutiva: {str(e)}")