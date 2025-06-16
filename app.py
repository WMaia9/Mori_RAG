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
def carregar_modelo_st():
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

# === 4. CARREGAMENTO E CONFIGURAÇÃO GLOBAL ===
modelo_st = carregar_modelo_st()
df_devolutivas = carregar_devolutivas()
df_rubricas = carregar_rubricas()

# --- Configurações na Barra Lateral ---
st.sidebar.title("Configurações de Teste")
st.sidebar.markdown("### 🔍 Modelo de Recomendação")
opcoes_modelo = [
    "Modelo Avançado (v2, Re-ranking)",
    "Modelo Intermediário (Busca Simples)",
    "Modelo Antigo (Legacy)"
]
modelo_ativo = st.sidebar.selectbox(
    "Escolha o motor de recomendação:",
    opcoes_modelo,
    index=0,
    help="Alterne entre os modelos para comparar a qualidade das recomendações."
)

# Carregamento condicional dos dados de busca com base na seleção
if modelo_ativo == "Modelo Antigo (Legacy)":
    st.sidebar.info("Base de dados original (stellav5). Lógica de busca simples.")
    index = carregar_index("data/odas/odas_index_stellav5.faiss")
    df_odas = carregar_metadados("data/odas/metadados_odas_stellav5.pkl")
elif modelo_ativo == "Modelo Intermediário (Busca Simples)":
    st.sidebar.info("Base de dados atualizada (1606), sem enriquecimento de IA. Lógica de busca simples.")
    index = carregar_index("data/odas/odas_index_1606.faiss")
    df_odas = carregar_metadados("data/odas/metadados_odas_1606.pkl")
else: # Modelo Avançado (v2, Re-ranking)
    st.sidebar.info("Base de dados enriquecida com IA (v2). Usa lógica de Re-ranking Ponderado.")
    index = carregar_index("data/odas/odas_index_1606_v2.faiss")
    df_odas = carregar_metadados("data/odas/metadados_odas_1606_v2.pkl")


# === 5. FUNÇÕES AUXILIARES ===
# ... (todas as suas funções auxiliares como encontrar_rubrica, gerar_card_material etc. permanecem aqui) ...
def encontrar_rubrica(pontuacao, dimensao, subdimensao):
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

def gerar_texto_devolutiva_markdown(pontuacao, dimensao, subdimensao):
    rubrica_numero, rubrica_nome, tipo_faixa = encontrar_rubrica(pontuacao, dimensao, subdimensao)
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
    return f"""
## 📄 Devolutiva Personalizada

- 🔢 **Pontuação:** {pontuacao}
- 📂 **Dimensão:** {dimensao}
- 📁 **Subdimensão:** {subdimensao}
- 🏷️ **Rubrica:** Rubrica {rubrica_numero} - {rubrica_nome}
- 📊 **Nível:** {tipo_faixa}

---

### ✅ Seus pontos fortes:

{item['Pontos fortes']}

---

### 📈 O que fazer para avançar:

{item['O que fazer para avançar']}

---

### 📚 Necessidades formativas:

{formatar_necessidades_formativas(item['Necessidades formativas'])}
""".strip()

def gerar_texto_devolutiva_rico(pontuacao, dimensao, subdimensao, modelo_selecionado):
    rubrica_numero, rubrica_nome, tipo_faixa = encontrar_rubrica(pontuacao, dimensao, subdimensao)
    if rubrica_numero is None or tipo_faixa is None: return None
    devolutiva = df_devolutivas[
        (df_devolutivas['Dimensão'] == dimensao) &
        (df_devolutivas['Subdimensão'] == subdimensao) &
        (df_devolutivas['Rubrica numero'] == rubrica_numero) &
        (df_devolutivas['Rubrica nome'] == f"{rubrica_nome} – Nível {tipo_faixa}")
    ]
    if devolutiva.empty: return None
    item = devolutiva.iloc[0]
    # A query só é enriquecida para o modelo avançado que entende esse contexto
    if modelo_selecionado == "Modelo Avançado (v2, Re-ranking)":
        contexto_query = f"Perfil do usuário: gestor no Nível {tipo_faixa} da Rubrica {rubrica_numero} - {rubrica_nome}. A necessidade de aprendizagem é a seguinte:"
        return f"{contexto_query}\n\n{item['Necessidades formativas']}".strip()
    else:
        return f"Necessidades formativas:\n{item['Necessidades formativas']}".strip()

def gerar_embedding_para_rag(texto: str) -> np.ndarray:
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

def obter_pontuacao_maxima(dimensao, subdimensao):
    rubricas_filtradas = df_rubricas[
        (df_rubricas['dimensao'] == dimensao) &
        (df_rubricas['subdimensao'] == subdimensao)
    ]
    if rubricas_filtradas.empty: return 51
    return int(rubricas_filtradas['faixa_total_max'].max())

# === 6. INTERFACE PRINCIPAL ===
st.title("📘 Geração de Devolutivas e Materiais Relacionados")
modo = st.radio("Escolha o modo:", ["Individual", "Geral"], key="modo_selecao")

# SUBSTITUA TODO O BLOCO if modo == "Individual": POR ESTE

if modo == "Individual":
    st.markdown("### Recomendação Individual")
    dimensao = st.selectbox("Escolha a dimensão:", sorted(df_devolutivas["Dimensão"].unique()))
    subdimensoes = df_devolutivas[df_devolutivas["Dimensão"] == dimensao]["Subdimensão"].unique()
    subdimensao = st.selectbox("Escolha a subdimensão:", sorted(subdimensoes))
    pontuacao_max = obter_pontuacao_maxima(dimensao, subdimensao)
    pontuacao = st.slider("Pontuação:", 0, pontuacao_max, min(17, pontuacao_max))

    if st.button("Gerar devolutiva e recomendações"):
        texto_markdown = gerar_texto_devolutiva_markdown(pontuacao, dimensao, subdimensao)
        texto_rico = gerar_texto_devolutiva_rico(pontuacao, dimensao, subdimensao, modelo_ativo)

        if texto_markdown is None or texto_rico is None:
            st.warning("Não foi possível gerar devolutiva para os dados informados.")
        else:
            st.markdown(texto_markdown)
            
            with st.spinner("Buscando e analisando os melhores materiais..."):

                # Gera o embedding da query uma única vez
                embedding_query = gerar_embedding_para_rag(texto_rico)

                # --- LÓGICA CONDICIONAL PRINCIPAL ---
                
                if modelo_ativo == "Modelo Avançado (v2, Re-ranking)":
                    # --- CAMINHO AVANÇADO ---
                    st.info("💡 Aplicando lógica Avançada (Filtro + Re-ranking)...")

                    # 1. Filtro Inteligente
                    rubrica_numero, rubrica_nome, _ = encontrar_rubrica(pontuacao, dimensao, subdimensao)
                    rubrica_alvo = f"Rubrica {rubrica_numero} - {rubrica_nome}" if rubrica_numero and rubrica_nome else None
                    
                    df_para_buscar = df_odas
                    if rubrica_alvo and 'Rubrica_IA' in df_odas.columns:
                        candidatos_filtro = df_odas[df_odas['Rubrica_IA'].str.contains(rubrica_alvo, na=False)]
                        if not candidatos_filtro.empty:
                            df_para_buscar = candidatos_filtro
                        else:
                            st.warning(f"Filtro não encontrou resultados para '{rubrica_alvo}'. Buscando na base completa.")
                    
                    indices_para_buscar = df_para_buscar.index.to_numpy()

                    # 2. Busca Vetorial e Interseção
                    k_busca = min(len(indices_para_buscar), 1000)
                    distancias, indices = index.search(embedding_query.astype("float32"), k=k_busca)
                    
                    resultados_finais = [(idx, dist) for idx, dist in zip(indices[0], distancias[0]) if idx in indices_para_buscar]
                    indices_finais = [r[0] for r in resultados_finais]
                    distancias_finais = [r[1] for r in resultados_finais]

                    resultados = df_odas.iloc[indices_finais].copy()
                    resultados["distância"] = distancias_finais
                    
                    # 3. Re-ranking Ponderado
                    if not resultados.empty:
                        resultados.rename(columns={'distância': 'score_semantico'}, inplace=True)
                        resultados['score_final'] = resultados['score_semantico']
                        BONUS_CONFIANCA_ALTA, PENALIDADE_CONFIANCA_BAIXA = 1.15, 0.90
                        if 'Confiança_IA' in resultados.columns:
                            for idx, row in resultados.iterrows():
                                confianca = str(row.get('Confiança_IA', '')).lower()
                                if 'alta' in confianca:
                                    resultados.loc[idx, 'score_final'] *= BONUS_CONFIANCA_ALTA
                                elif 'baixa' in confianca:
                                    resultados.loc[idx, 'score_final'] *= PENALIDADE_CONFIANCA_BAIXA
                        resultados = resultados.sort_values(by='score_final', ascending=False)
                        resultados.rename(columns={'score_final': 'distância'}, inplace=True)

                else:
                    # --- CAMINHO SIMPLES (PARA MODELO INTERMEDIÁRIO E ANTIGO) ---
                    st.info(f"ℹ️ Aplicando lógica de Busca Simples para o {modelo_ativo}...")
                    
                    # Busca ampla na base de dados inteira, sem filtros prévios
                    k_busca = 1000 
                    distancias, indices = index.search(embedding_query.astype("float32"), k=k_busca)
                    
                    resultados = df_odas.iloc[indices[0]].copy()
                    resultados["distância"] = distancias[0]

                # --- ETAPA FINAL COMUM A TODOS: BALANCEAMENTO E EXIBIÇÃO ---
                
                st.info("Balanceando os tipos de materiais encontrados...")
                # ... (código de balanceamento com np.select e groupby) ...
                condicoes = [
                    resultados['Suporte'].str.contains(r"jogo|painel", case=False, na=False),
                    resultados['Suporte'].str.contains(r"infográfico|mapa|tabela|gráfico|slide", case=False, na=False),
                    resultados['Suporte'].str.contains(r"vídeo|video|curso|aula|aula gravada|palestra|webinário|animação|exposição", case=False, na=False),
                    resultados['Suporte'].str.contains(r"áudio|audio|podcast|rádio|entrevista", case=False, na=False),
                    resultados['Suporte'].str.contains(r"texto|artigo|livro|relatório|resenha|plano de aula|documento institucional|manual|guia|tutorial|documento oficial|documento técnico|cartilha|blog|apostila|coletânea|recomendação", case=False, na=False)
                ]
                categorias = ["interativos", "visuais", "videos", "audios", "artigos"]
                resultados['categoria'] = np.select(condicoes, categorias, default='outros')
                resultados_categorizados = resultados[resultados['categoria'] != 'outros']
                resultados_balanceados = resultados_categorizados.groupby('categoria').head(10)
                
                artigos = resultados_balanceados[resultados_balanceados['categoria'] == 'artigos']
                videos = resultados_balanceados[resultados_balanceados['categoria'] == 'videos']
                audios = resultados_balanceados[resultados_balanceados['categoria'] == 'audios']
                visuais = resultados_balanceados[resultados_balanceados['categoria'] == 'visuais']
                interativos = resultados_balanceados[resultados_balanceados['categoria'] == 'interativos']
                
                titulo_secao = f"do {modelo_ativo}"
                todas_listas = [artigos, videos, audios, visuais, interativos]

                if any(not df.empty for df in todas_listas):
                    st.markdown(f"--- \n### Materiais Recomendados ({titulo_secao})")
                    if not artigos.empty:
                        st.markdown("#### 📚 Textos e Artigos")
                        for i, row in enumerate(artigos.itertuples()): st.markdown(gerar_card_material(row._asdict(), i))
                    if not videos.empty:
                        st.markdown("#### 🎥 Vídeos e Aulas")
                        for i, row in enumerate(videos.itertuples()): st.markdown(gerar_card_material(row._asdict(), i))
                    if not audios.empty:
                        st.markdown("#### 🎧 Áudios e Podcasts")
                        for i, row in enumerate(audios.itertuples()): st.markdown(gerar_card_material(row._asdict(), i))
                    if not visuais.empty:
                        st.markdown("#### 📊 Materiais Visuais")
                        for i, row in enumerate(visuais.itertuples()): st.markdown(gerar_card_material(row._asdict(), i))
                    if not interativos.empty:
                        st.markdown("#### 🎮 Materiais Interativos")
                        for i, row in enumerate(interativos.itertuples()): st.markdown(gerar_card_material(row._asdict(), i))
                else:
                    st.info("Nenhum material encontrado para esta combinação.")

elif modo == "Geral":
    st.markdown("### Devolutiva Geral da Dimensão")
    # A configuração do modelo GPT para síntese permanece na sidebar
    st.sidebar.markdown("### 🤖 Configurações de IA (Síntese)")
    modelo_gpt_selecionado = st.sidebar.selectbox(
        "Escolha o modelo de IA:",
        ["gpt-4o-mini", "gpt-4"], index=0,
        help="Usado para gerar o texto de síntese no Modo Geral."
    )
    
    dimensao_escolhida = st.selectbox("Escolha a dimensão que deseja gerar a devolutiva geral:", ["Planejamento pedagógico", "Pessoal-relacional"])

    if dimensao_escolhida == "Planejamento pedagógico":
        st.markdown("#### Informe as pontuações das subdimensões pedagógicas:")
        subdimensoes = [
            "Desenvolvimento profissional docente", "Implementação do processo de ensino e aprendizagem",
            "Monitoramento e Avaliação da Aprendizagem", "Planejamento pedagógico", "Proteção das Trajetórias Estudantis"
        ]
        pontuacoes = {}
        for sub in subdimensoes:
            max_ponto = int(df_rubricas[df_rubricas['subdimensao'] == sub]['faixa_total_max'].max())
            pontuacoes[sub] = st.slider(f"{sub}", 0, max_ponto, 0, key=f"slider_{sub}")

        openai_api_key = st.text_input("Insira sua OpenAI API Key para a síntese", type="password", key="geral_api_key_1")

        if st.button("Gerar devolutiva da dimensão pedagógica") and openai_api_key:
            partes = [gerar_texto_devolutiva_rico(ponto, "Dimensão pedagógica", sub) for sub, ponto in pontuacoes.items()]
            partes_validas = [p for p in partes if p]

            if not partes_validas:
                st.warning("⚠️ Nenhuma pontuação informada ou devolutiva encontrada.")
            else:
                prompt = f"Você é um assistente especializado em gestão escolar. Seu objetivo é receber as devolutivas textuais de cada subdimensão e gerar um texto síntese único para a dimensão “Planejamento pedagógico”.\n\nTarefa:\n- Identificar e sintetizar os principais pontos fortes que emergem de todas as subdimensões.\n- Apontar as ações concretas que o gestor deve implementar para avançar ao próximo nível de maturidade (conforme rubricas).\n- Limite: até 3 parágrafos.\n- Tom: claro, direto, orientado a “próximos passos”.\n\n---\n{chr(10).join(partes_validas)}"
                try:
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model=modelo_gpt_selecionado,
                        messages=[{"role": "system", "content": "Você é um especialista em formação de professores."}, {"role": "user", "content": prompt}],
                        temperature=0.7, max_tokens=1500
                    )
                    st.markdown("### 📖 Devolutiva da Dimensão Pedagógica")
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Erro ao gerar devolutiva: {str(e)}")

    elif dimensao_escolhida == "Pessoal-relacional":
        st.markdown("#### Informe a pontuação da subdimensão:")
        sub = "Convivência no ambiente escolar"
        max_ponto = int(df_rubricas[df_rubricas['subdimensao'] == sub]['faixa_total_max'].max())
        ponto = st.slider(f"{sub}", 0, max_ponto, 0, key=f"slider_{sub}")

        openai_api_key = st.text_input("Insira sua OpenAI API Key para a síntese", type="password", key="geral_api_key_2")

        if st.button("Gerar devolutiva da dimensão pessoal-relacional") and openai_api_key:
            texto = gerar_texto_devolutiva_rico(ponto, "Dimensão pessoal-relacional", sub)
            if not texto:
                st.warning("⚠️ Nenhuma pontuação informada.")
            else:
                prompt = f"Você é um assistente especializado em gestão escolar. Seu objetivo é receber as devolutivas textuais de cada subdimensão e gerar um texto síntese único para a dimensão “Pessoal-Relacional”.\n\nTarefa:\n- Identificar e sintetizar os principais pontos fortes que emergem da subdimensão.\n- Apontar as ações concretas que o gestor deve implementar para avançar ao próximo nível de maturidade (conforme rubricas).\n- Limite: até 3 parágrafos.\n- Tom: claro, direto, orientado a “próximos passos”.\n\nSubdimensão {sub}:\n{texto}"
                try:
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model=modelo_gpt_selecionado,
                        messages=[{"role": "system", "content": "Você é um especialista em formação de professores."}, {"role": "user", "content": prompt}],
                        temperature=0.7, max_tokens=1000
                    )
                    st.markdown("### 📖 Devolutiva da Dimensão Pessoal-Relacional")
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Erro ao gerar devolutiva: {str(e)}")