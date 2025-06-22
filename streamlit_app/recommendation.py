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
) -> Tuple[pd.DataFrame, ...]:
    """
    Orquestra o processo de busca, que pode ser simples ou uma combinação de
    filtragem, re-ranking e preenchimento híbrido para o modelo avançado.
    """
    
    # --- ETAPA 1: BUSCA VETORIAL AMPLA (BASE PARA TODAS AS LÓGICAS) ---
    embedding_query = gerar_embedding_para_rag(modelo_st, texto_rico)
    # Fazemos uma busca ampla para ter muitos candidatos para ambas as lógicas
    k_busca = 1000 
    distancias, indices = index.search(embedding_query.astype("float32"), k=k_busca)
    
    # Cria um DataFrame base com os resultados da busca ampla
    resultados_gerais = df_odas.iloc[indices[0]].copy()
    resultados_gerais["distância"] = distancias[0]

    # --- ETAPA 2: LÓGICA CONDICIONAL ---
    
    if modelo_ativo == "Modelo Avançado (v2, Re-ranking)":
        # --- LÓGICA AVANÇADA COM PREENCHIMENTO HÍBRIDO ---
        st.info("💡 Aplicando lógica Avançada (Precisão + Preenchimento)...")

        # 2a. Camada de Precisão: Filtra e Re-rankeia os melhores
        rubrica_numero, rubrica_nome, _ = encontrar_rubrica(df_rubricas, pontuacao, dimensao, subdimensao)
        rubrica_alvo = rubrica_nome if rubrica_nome else None

        resultados_precisos = pd.DataFrame()
        if rubrica_alvo and 'Rubrica_IA' in resultados_gerais.columns:
            candidatos_filtro = resultados_gerais[resultados_gerais['Rubrica_IA'].str.contains(rubrica_alvo, na=False, case=False)]
            if not candidatos_filtro.empty:
                # Re-ranking Ponderado apenas nos candidatos do filtro
                candidatos_filtro = candidatos_filtro.copy()
                candidatos_filtro.rename(columns={'distância': 'score_semantico'}, inplace=True)
                candidatos_filtro['score_final'] = candidatos_filtro['score_semantico']
                BONUS_CONFIANCA_ALTA, PENALIDADE_CONFIANCA_BAIXA = 1.15, 0.90
                
                for idx, row in candidatos_filtro.iterrows():
                    confianca = str(row.get('Confiança_IA', '')).lower()
                    if 'alta' in confianca:
                        candidatos_filtro.loc[idx, 'score_final'] *= BONUS_CONFIANCA_ALTA
                    elif 'baixa' in confianca:
                        candidatos_filtro.loc[idx, 'score_final'] *= PENALIDADE_CONFIANCA_BAIXA
                
                resultados_precisos = candidatos_filtro.sort_values(by='score_final', ascending=False)
                resultados_precisos.rename(columns={'score_final': 'distância'}, inplace=True)

        # 2b. Lógica de Preenchimento Híbrido
        st.info("Balanceando os tipos de materiais (Precisão + Preenchimento)...")
        
        categorias_regex = {
            "interativos": r"jogo|painel",
            "visuais": r"infográfico|mapa|tabela|gráfico|slide",
            "videos": r"vídeo|video|curso|aula|aula gravada|palestra|webinário|animação|exposição",
            "audios": r"áudio|audio|podcast|rádio|entrevista",
            "artigos": r"texto|artigo|livro|relatório|resenha|plano de aula|documento institucional|manual|guia|tutorial|documento oficial|documento técnico|cartilha|blog|apostila|coletânea|recomendação"
        }
        
        listas_finais = {cat: [] for cat in categorias_regex.keys()}
        ids_ja_adicionados = set()
        QUOTA = 10

        # Primeira passagem: preenche com os resultados de precisão
        if not resultados_precisos.empty:
            for index, row in resultados_precisos.iterrows():
                for categoria, regex in categorias_regex.items():
                    if len(listas_finais[categoria]) < QUOTA and re.search(regex, str(row.get("Suporte", "")).lower()):
                        listas_finais[categoria].append(row)
                        ids_ja_adicionados.add(index)
                        break # Vai para a próxima linha do DataFrame

        # Segunda passagem: completa as cotas com os resultados da busca geral
        for index, row in resultados_gerais.iterrows():
            if index in ids_ja_adicionados:
                continue
            
            for categoria, regex in categorias_regex.items():
                if len(listas_finais[categoria]) < QUOTA and re.search(regex, str(row.get("Suporte", "")).lower()):
                    listas_finais[categoria].append(row)
                    ids_ja_adicionados.add(index)
                    break
        
        # Converte as listas de Series em DataFrames
        artigos = pd.DataFrame(listas_finais["artigos"]) if listas_finais["artigos"] else pd.DataFrame()
        videos = pd.DataFrame(listas_finais["videos"]) if listas_finais["videos"] else pd.DataFrame()
        audios = pd.DataFrame(listas_finais["audios"]) if listas_finais["audios"] else pd.DataFrame()
        visuais = pd.DataFrame(listas_finais["visuais"]) if listas_finais["visuais"] else pd.DataFrame()
        interativos = pd.DataFrame(listas_finais["interativos"]) if listas_finais["interativos"] else pd.DataFrame()

        return artigos, videos, audios, visuais, interativos

    else:
        # --- LÓGICA SIMPLES (PARA MODELO INTERMEDIÁRIO E ANTIGO) ---
        st.info(f"ℹ️ Aplicando lógica de Busca Simples para o {modelo_ativo}...")
        
        # A lógica de balanceamento otimizada é aplicada diretamente nos resultados da busca ampla
        condicoes = [
            resultados_gerais['Suporte'].str.contains(r"jogo|painel", case=False, na=False),
            resultados_gerais['Suporte'].str.contains(r"infográfico|mapa|tabela|gráfico|slide", case=False, na=False),
            resultados_gerais['Suporte'].str.contains(r"vídeo|video|curso|aula|aula gravada|palestra|webinário|animação|exposição", case=False, na=False),
            resultados_gerais['Suporte'].str.contains(r"áudio|audio|podcast|rádio|entrevista", case=False, na=False),
            resultados_gerais['Suporte'].str.contains(r"texto|artigo|livro|relatório|resenha|plano de aula|documento institucional|manual|guia|tutorial|documento oficial|documento técnico|cartilha|blog|apostila|coletânea|recomendação", case=False, na=False)
        ]
        categorias = ["interativos", "visuais", "videos", "audios", "artigos"]
        resultados_gerais['categoria'] = np.select(condicoes, categorias, default='outros')
        
        resultados_categorizados = resultados_gerais[resultados_gerais['categoria'] != 'outros']
        resultados_balanceados = resultados_categorizados.groupby('categoria').head(10)
        
        artigos = resultados_balanceados[resultados_balanceados['categoria'] == 'artigos']
        videos = resultados_balanceados[resultados_balanceados['categoria'] == 'videos']
        audios = resultados_balanceados[resultados_balanceados['categoria'] == 'audios']
        visuais = resultados_balanceados[resultados_balanceados['categoria'] == 'visuais']
        interativos = resultados_balanceados[resultados_balanceados['categoria'] == 'interativos']
        
        return artigos, videos, audios, visuais, interativos