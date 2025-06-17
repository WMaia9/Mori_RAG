# app.py
# Arquivo principal que controla a interface e orquestra as chamadas
# para os módulos de utilidades e de recomendação.

import streamlit as st
from openai import OpenAI
from typing import Any, Dict
from utils import *
from recommendation import get_recommendations

# === CONFIGURAÇÃO E CARREGAMENTO GLOBAL ===
st.set_page_config(page_title="📘 Geração de Devolutivas e Materiais", layout="wide")

# As funções de carregamento são chamadas a partir de utils.py
modelo_st = carregar_modelo_st()
df_devolutivas = carregar_devolutivas()
df_rubricas = carregar_rubricas()

# --- Configurações na Barra Lateral ---
st.sidebar.title("Configurações")
st.sidebar.markdown("### 🔍 Modelo de Busca")
modelo_ativo = st.sidebar.selectbox(
    "Escolha o motor de recomendação:",
    ["Modelo Avançado (v2, Re-ranking)", "Modelo Intermediário (Busca Simples)", "Modelo Antigo (Legacy)"],
    index=0,
    help="Alterne entre os modelos para comparar a qualidade das recomendações."
)

# Carregamento condicional dos dados de busca
if modelo_ativo == "Modelo Antigo (Legacy)":
    st.sidebar.info("Base de dados original (stellav5).")
    index = carregar_index("data/odas/odas_index_stellav5.faiss")
    df_odas = carregar_metadados("data/odas/metadados_odas_stellav5.pkl")
elif modelo_ativo == "Modelo Intermediário (Busca Simples)":
    st.sidebar.info("Base de dados atualizada (1606), busca simples.")
    index = carregar_index("data/odas/odas_index_1606.faiss")
    df_odas = carregar_metadados("data/odas/metadados_odas_1606.pkl")
else: # Modelo Avançado
    st.sidebar.info("Base de dados enriquecida com IA (v2). Lógica de Re-ranking.")
    index = carregar_index("data/odas/odas_index_1606_v2.faiss")
    df_odas = carregar_metadados("data/odas/metadados_odas_1606_v2.pkl")


# === INTERFACE PRINCIPAL ===
st.title("📘 Geração de Devolutivas e Materiais Relacionados")
modo = st.radio("Escolha o modo:", ["Individual", "Geral"], key="modo_selecao")

if modo == "Individual":
    st.markdown("### Recomendação Individual")
    
    # --- Coleta de Input do Usuário ---
    dimensao = st.selectbox("Escolha a dimensão:", sorted(df_devolutivas["Dimensão"].unique()))
    subdimensoes = sorted(df_devolutivas[df_devolutivas["Dimensão"] == dimensao]["Subdimensão"].unique())
    subdimensao = st.selectbox("Escolha a subdimensão:", subdimensoes)
    pontuacao_max = obter_pontuacao_maxima(df_rubricas, dimensao, subdimensao)
    pontuacao = st.slider("Pontuação:", 0, pontuacao_max, min(17, pontuacao_max))

    if st.button("Gerar devolutiva e recomendações"):
        # --- Geração da Devolutiva ---
        texto_markdown = gerar_texto_devolutiva_markdown(df_devolutivas, df_rubricas, pontuacao, dimensao, subdimensao)
        
        if texto_markdown is None:
            st.warning("Não foi possível gerar devolutiva para os dados informados.")
        else:
            st.markdown(texto_markdown)
            
            with st.spinner("Buscando e analisando os melhores materiais..."):
                texto_rico = gerar_texto_devolutiva_rico(df_devolutivas, df_rubricas, pontuacao, dimensao, subdimensao, modelo_ativo)
                
                # A mágica acontece aqui: chamamos nossa função centralizada do recommendation.py
                artigos, videos, audios, visuais, interativos = get_recommendations(
                    modelo_st, index, df_odas, df_rubricas, pontuacao, dimensao, subdimensao, texto_rico, modelo_ativo
                )

            # --- Exibição dos Resultados ---
            titulo_secao = f"do {modelo_ativo}"
            todas_listas = [artigos, videos, audios, visuais, interativos]

            if any(not df.empty for df in todas_listas):
                st.markdown(f"--- \n### Materiais Recomendados ({titulo_secao})")
                
                # Função auxiliar para não repetir o código de exibição
                def exibir_categoria(titulo: str, emoji: str, df: pd.DataFrame):
                    if not df.empty:
                        st.markdown(f"#### {emoji} {titulo}")
                        for i, row in enumerate(df.itertuples()):
                            st.markdown(gerar_card_material(row._asdict(), i))

                exibir_categoria("Textos e Artigos", "📚", artigos)
                exibir_categoria("Vídeos e Aulas", "🎥", videos)
                exibir_categoria("Áudios e Podcasts", "🎧", audios)
                exibir_categoria("Materiais Visuais", "📊", visuais)
                exibir_categoria("Materiais Interativos", "🎮", interativos)
            else:
                st.info("Nenhum material relevante encontrado para esta combinação.")

elif modo == "Geral":
    st.markdown("### Devolutiva Geral da Dimensão")
    
    st.sidebar.markdown("### 🤖 Configurações de IA (Síntese)")
    modelo_gpt_selecionado = st.sidebar.selectbox(
        "Escolha o modelo de IA:", ["gpt-4o-mini", "gpt-4"], index=0,
        help="Usado para gerar o texto de síntese no Modo Geral."
    )
    
    dimensao_escolhida = st.selectbox("Escolha a dimensão para gerar a devolutiva geral:", ["Planejamento pedagógico", "Pessoal-relacional"])

    # --- Lógica para a Dimensão "Planejamento pedagógico" ---
    if dimensao_escolhida == "Planejamento pedagógico":
        st.markdown("#### Informe as pontuações das subdimensões pedagógicas:")
        subdimensoes = [
            "Desenvolvimento profissional docente", "Implementação do processo de ensino e aprendizagem",
            "Monitoramento e Avaliação da Aprendizagem", "Planejamento pedagógico", "Proteção das Trajetórias Estudantis"
        ]
        pontuacoes = {}
        for sub in subdimensoes:
            max_ponto = obter_pontuacao_maxima(df_rubricas, "Dimensão pedagógica", sub)
            pontuacoes[sub] = st.slider(f"{sub}", 0, max_ponto, 0, key=f"slider_{sub}")

        openai_api_key = st.text_input("Insira sua OpenAI API Key para a síntese", type="password", key="geral_api_key_1")

        if st.button("Gerar devolutiva da dimensão pedagógica") and openai_api_key:
            partes = [gerar_texto_devolutiva_rico(df_devolutivas, df_rubricas, ponto, "Dimensão pedagógica", sub, modelo_ativo) for sub, ponto in pontuacoes.items()]
            partes_validas = [p for p in partes if p]

            if not partes_validas:
                st.warning("⚠️ Nenhuma pontuação informada ou devolutiva encontrada.")
            else:
                prompt = f"Você é um assistente especializado em gestão escolar. Seu objetivo é receber as devolutivas textuais de cada subdimensão e gerar um texto síntese único para a dimensão “Planejamento pedagógico”.\n\nTarefa:\n- Identificar e sintetizar os principais pontos fortes que emergem de todas as subdimensões.\n- Apontar as ações concretas que o gestor deve implementar para avançar ao próximo nível de maturidade (conforme rubricas).\n- Limite: até 3 parágrafos.\n- Tom: claro, direto, orientado a “próximos passos”.\n\n---\n{chr(10).join(partes_validas)}"
                
                client = OpenAI(api_key=openai_api_key)
                with st.spinner("Gerando síntese com a IA..."):
                    resposta_sintetizada = sintetizar_devolutiva_com_ia(client, modelo_gpt_selecionado, prompt, max_tokens=1500)
                
                if resposta_sintetizada:
                    st.markdown("### 📖 Devolutiva da Dimensão Pedagógica")
                    st.markdown(resposta_sintetizada)

    # --- Lógica para a Dimensão "Pessoal-relacional" ---
    elif dimensao_escolhida == "Pessoal-relacional":
        st.markdown("#### Informe a pontuação da subdimensão:")
        sub = "Convivência no ambiente escolar"
        max_ponto = obter_pontuacao_maxima(df_rubricas, "Dimensão pessoal-relacional", sub)
        ponto = st.slider(f"{sub}", 0, max_ponto, 0, key=f"slider_{sub}")

        openai_api_key = st.text_input("Insira sua OpenAI API Key para a síntese", type="password", key="geral_api_key_2")

        if st.button("Gerar devolutiva da dimensão pessoal-relacional") and openai_api_key:
            texto = gerar_texto_devolutiva_rico(df_devolutivas, df_rubricas, ponto, "Dimensão pessoal-relacional", sub, modelo_ativo)
            if not texto:
                st.warning("⚠️ Nenhuma pontuação informada.")
            else:
                prompt = f"Você é um assistente especializado em gestão escolar. Seu objetivo é receber as devolutivas textuais de cada subdimensão e gerar um texto síntese único para a dimensão “Pessoal-Relacional”.\n\nTarefa:\n- Identificar e sintetizar os principais pontos fortes que emergem da subdimensão.\n- Apontar as ações concretas que o gestor deve implementar para avançar ao próximo nível de maturidade (conforme rubricas).\n- Limite: até 3 parágrafos.\n- Tom: claro, direto, orientado a “próximos passos”.\n\nSubdimensão {sub}:\n{texto}"
                
                client = OpenAI(api_key=openai_api_key)
                with st.spinner("Gerando síntese com a IA..."):
                    resposta_sintetizada = sintetizar_devolutiva_com_ia(client, modelo_gpt_selecionado, prompt, max_tokens=1000)

                if resposta_sintetizada:
                    st.markdown("### 📖 Devolutiva da Dimensão Pessoal-Relacional")
                    st.markdown(resposta_sintetizada)