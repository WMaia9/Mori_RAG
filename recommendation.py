# recommendation.py
# Este módulo contém a lógica central do motor de recomendação,
# orquestrando a busca, o re-ranking e o balanceamento.

import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Tuple
from utils import gerar_embedding_para_rag, encontrar_rubrica

def get_recommendations(
    modelo_st, 
    index, 
    df_odas: pd.DataFrame, 
    df_rubricas: pd.DataFrame, 
    pontuacao: int, 
    dimensao: str, 
    subdimensao: str, 
    texto_rico: str, 
    modelo_ativo: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executa o pipeline completo de recomendação.

    Args:
        modelo_st: O modelo de embedding carregado.
        index: O índice FAISS carregado.
        df_odas (pd.DataFrame): DataFrame com os metadados dos materiais.
        df_rubricas (pd.DataFrame): DataFrame com as regras das rubricas.
        pontuacao (int): Pontuação do usuário.
        dimensao (str): Dimensão avaliada.
        subdimensao (str): Subdimensão avaliada.
        texto_rico (str): O texto-query gerado pela devolutiva.
        modelo_ativo (str): O nome do motor de recomendação selecionado.

    Returns:
        Tuple[pd.DataFrame, ...]: Uma tupla de 5 DataFrames, um para cada categoria
                                  de material, já balanceados e ordenados.
    """
    
    # --- ETAPA 1: BUSCA E RE-RANKING CONDICIONAL ---
    
    embedding_query = gerar_embedding_para_rag(modelo_st, texto_rico)
    
    if modelo_ativo == "Modelo Avançado (v2, Re-ranking)":
        # LÓGICA AVANÇADA: Filtro -> Busca -> Interseção -> Re-ranking
        st.info("💡 Aplicando lógica Avançada (Filtro + Re-ranking)...")

        # 1a. Filtro Inteligente
        rubrica_numero, rubrica_nome, _ = encontrar_rubrica(df_rubricas, pontuacao, dimensao, subdimensao)
        rubrica_alvo = f"Rubrica {rubrica_numero} - {rubrica_nome}" if rubrica_numero and rubrica_nome else None
        
        df_para_buscar = df_odas
        if rubrica_alvo and 'Rubrica_IA' in df_odas.columns:
            candidatos_filtro = df_odas[df_odas['Rubrica_IA'].str.contains(rubrica_alvo, na=False)]
            if not candidatos_filtro.empty:
                df_para_buscar = candidatos_filtro
            else:
                st.warning(f"Filtro não encontrou resultados para '{rubrica_alvo}'. Buscando na base completa.")
        
        indices_para_buscar = df_para_buscar.index.to_numpy()

        # 1b. Busca Vetorial e Interseção
        k_busca = min(len(indices_para_buscar), 1000)
        distancias, indices = index.search(embedding_query.astype("float32"), k=k_busca)
        
        resultados_finais = [(idx, dist) for idx, dist in zip(indices[0], distancias[0]) if idx in indices_para_buscar]
        
        if not resultados_finais:
             return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        indices_finais = [r[0] for r in resultados_finais]
        distancias_finais = [r[1] for r in resultados_finais]
        
        resultados = df_odas.iloc[indices_finais].copy()
        resultados["distância"] = distancias_finais
        
        # 1c. Re-ranking Ponderado
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
        # LÓGICA SIMPLES: Busca ampla na base de dados inteira
        st.info(f"ℹ️ Aplicando lógica de Busca Simples para o {modelo_ativo}...")
        k_busca = 1000 
        distancias, indices = index.search(embedding_query.astype("float32"), k=k_busca)
        
        resultados = df_odas.iloc[indices[0]].copy()
        resultados["distância"] = distancias[0]

    # --- ETAPA 2: BALANCEAMENTO DE COTAS (COMUM A TODOS) ---
    st.info("Balanceando os tipos de materiais encontrados...")
    
    # Regex para cada categoria
    condicoes = [
        resultados['Suporte'].str.contains(r"jogo|painel", case=False, na=False),
        resultados['Suporte'].str.contains(r"infográfico|mapa|tabela|gráfico|slide", case=False, na=False),
        resultados['Suporte'].str.contains(r"vídeo|video|curso|aula|aula gravada|palestra|webinário|animação|exposição", case=False, na=False),
        resultados['Suporte'].str.contains(r"áudio|audio|podcast|rádio|entrevista", case=False, na=False),
        resultados['Suporte'].str.contains(r"texto|artigo|livro|relatório|resenha|plano de aula|documento institucional|manual|guia|tutorial|documento oficial|documento técnico|cartilha|blog|apostila|coletânea|recomendação", case=False, na=False)
    ]
    categorias = ["interativos", "visuais", "videos", "audios", "artigos"]
    
    # Classificação vetorizada
    resultados['categoria'] = np.select(condicoes, categorias, default='outros')
    
    # Agrupamento para garantir as cotas
    resultados_categorizados = resultados[resultados['categoria'] != 'outros']
    resultados_balanceados = resultados_categorizados.groupby('categoria').head(10)
    
    # Separação final
    artigos = resultados_balanceados[resultados_balanceados['categoria'] == 'artigos']
    videos = resultados_balanceados[resultados_balanceados['categoria'] == 'videos']
    audios = resultados_balanceados[resultados_balanceados['categoria'] == 'audios']
    visuais = resultados_balanceados[resultados_balanceados['categoria'] == 'visuais']
    interativos = resultados_balanceados[resultados_balanceados['categoria'] == 'interativos']

    return artigos, videos, audios, visuais, interativos