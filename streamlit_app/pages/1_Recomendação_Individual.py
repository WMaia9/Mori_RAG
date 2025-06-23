# pages/1_Recomendacao_Individual.py

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

app_root = str(Path(__file__).parent.parent)
if app_root not in sys.path:
    sys.path.append(app_root)

from src.utils import *
from src.recommendation import get_recommendations

# --- CABEÇALHO ---
exibir_cabecalho()
st.markdown("### Recomendação Individual")

# --- SELEÇÃO DO MOTOR DE RECOMENDAÇÃO ---
st.markdown("#### 1. Escolha o Motor de Recomendação")
modelo_ativo = st.selectbox(
    "Selecione o motor de recomendação que deseja testar:",
    ["Modelo Avançado (v2, Re-ranking)", "Modelo Intermediário (Busca Simples)", "Modelo Antigo (Legacy)"],
    index=0,
    help="Alterne entre os modelos para comparar a qualidade das recomendações."
)

# --- CARREGAMENTO DE DADOS ---
with st.spinner(f"Carregando dados para o {modelo_ativo}..."):
    if modelo_ativo == "Modelo Antigo (Legacy)":
        index = carregar_index("models/odas_index_stellav5.faiss")
        df_odas = carregar_metadados("models/metadados_odas_stellav5.pkl")
    elif modelo_ativo == "Modelo Intermediário (Busca Simples)":
        index = carregar_index("models/odas_index_1606.faiss")
        df_odas = carregar_metadados("models/metadados_odas_1606.pkl")
    else: # Modelo Avançado
        index = carregar_index("models/odas_index_1606_v2.faiss")
        df_odas = carregar_metadados("models/metadados_odas_1606_v2.pkl")

# Carrega os dataframes que são sempre necessários
modelo_st = carregar_modelo_st()
df_devolutivas = carregar_devolutivas()
df_rubricas = carregar_rubricas()
st.success("Dados carregados!")

# --- INTERFACE DA PÁGINA ---
st.markdown("---")
st.markdown("#### 2. Preencha os dados para a recomendação")
dimensao = st.selectbox("Escolha a dimensão:", sorted(df_devolutivas["Dimensão"].unique()))
subdimensoes = sorted(df_devolutivas[df_devolutivas["Dimensão"] == dimensao]["Subdimensão"].unique())
subdimensao = st.selectbox("Escolha a subdimensão:", subdimensoes)
pontuacao_max = obter_pontuacao_maxima(df_rubricas, dimensao, subdimensao)
pontuacao = st.slider("Pontuação:", 0, pontuacao_max, min(17, pontuacao_max))

if st.button("Gerar devolutiva e recomendações"):
    texto_markdown = gerar_texto_devolutiva_markdown(df_devolutivas, df_rubricas, pontuacao, dimensao, subdimensao)
    
    if texto_markdown is None:
        st.warning("Não foi possível gerar devolutiva para os dados informados.")
    else:
        st.markdown(texto_markdown)
        
        with st.spinner(f"Buscando recomendações com o {modelo_ativo}..."):
            texto_rico = gerar_texto_devolutiva_rico(df_devolutivas, df_rubricas, pontuacao, dimensao, subdimensao, modelo_ativo)
            
            if texto_rico:
                artigos, videos, audios, visuais, interativos = get_recommendations(
                    modelo_st, index, df_odas, df_rubricas, pontuacao, dimensao, subdimensao, texto_rico, modelo_ativo
                )
            else:
                artigos, videos, audios, visuais, interativos = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        # Exibição dos Resultados
        st.markdown("---")
        st.header("Materiais Recomendados")
        st.info(f"Resultados gerados pelo **{modelo_ativo}**.")
        todas_listas = [artigos, videos, audios, visuais, interativos]

        if any(not df.empty for df in todas_listas):
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