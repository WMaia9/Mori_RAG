# recommendation.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from utils import gerar_embedding_para_rag, encontrar_rubrica # Importa funções do nosso outro arquivo

def get_recommendations(modelo_st, index, df_odas, df_rubricas, pontuacao, dimensao, subdimensao, texto_rico, modelo_ativo):
    """
    Orquestra o processo de busca, re-ranking e balanceamento de materiais.
    """
    # ETAPA 1: FILTRO INTELIGENTE (APENAS PARA MODELO AVANÇADO)
    df_para_buscar = df_odas
    if modelo_ativo == "Modelo Avançado (v2, Re-ranking)":
        rubrica_numero, rubrica_nome, _ = encontrar_rubrica(df_rubricas, pontuacao, dimensao, subdimensao)
        rubrica_alvo = f"Rubrica {rubrica_numero} - {rubrica_nome}" if rubrica_numero and rubrica_nome else None
        
        if rubrica_alvo and 'Rubrica_IA' in df_odas.columns:
            st.info(f"Aplicando filtro inteligente para: **{rubrica_alvo}**")
            candidatos_filtro = df_odas[df_odas['Rubrica_IA'].str.contains(rubrica_alvo, na=False)]
            if not candidatos_filtro.empty:
                df_para_buscar = candidatos_filtro
            else:
                st.warning(f"Filtro não encontrou resultados para '{rubrica_alvo}'. Buscando na base completa.")
    
    indices_para_buscar = df_para_buscar.index.to_numpy()

    # ETAPA 2: BUSCA VETORIAL E INTERSEÇÃO
    embedding_query = gerar_embedding_para_rag(modelo_st, texto_rico)
    k_busca = min(len(indices_para_buscar), 1000)
    distancias, indices = index.search(embedding_query.astype("float32"), k=k_busca)
    
    resultados_finais = [(idx, dist) for idx, dist in zip(indices[0], distancias[0]) if idx in indices_para_buscar]
    indices_finais = [r[0] for r in resultados_finais]
    distancias_finais = [r[1] for r in resultados_finais]

    if not indices_finais: # Se a interseção for vazia, retorna DataFrames vazios
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    resultados = df_odas.iloc[indices_finais].copy()
    resultados["distância"] = distancias_finais

    # ETAPA 3: RE-RANKING PONDERADO (APENAS PARA O MODELO AVANÇADO)
    if modelo_ativo == "Modelo Avançado (v2, Re-ranking)":
        st.info("💡 Aplicando lógica de Re-ranking Ponderado...")
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
    
    # ETAPA 4: BALANCEAMENTO DE COTAS
    st.info("Balanceando os tipos de materiais encontrados...")
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

    return artigos, videos, audios, visuais, interativos