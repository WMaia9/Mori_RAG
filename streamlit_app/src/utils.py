# utils.py
# Este módulo contém todas as funções auxiliares para carregamento de dados,
# processamento de texto e geração de componentes visuais.

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import Tuple, Dict, Any, Optional
from pathlib import Path # Importa a biblioteca para manipulação de caminhos

# === DEFINIÇÃO DA RAIZ DO APP ===
# Esta linha encontra o caminho para a pasta 'src' e sobe um nível para chegar na raiz 'streamlit_app'
APP_ROOT = Path(__file__).parent.parent

# === FUNÇÕES DE CACHE COM CAMINHOS CORRIGIDOS ===

@st.cache_resource
def carregar_modelo_st() -> SentenceTransformer:
    """Carrega o modelo de embedding SentenceTransformer."""
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

@st.cache_resource
def carregar_index(caminho_relativo: str) -> faiss.Index:
    """Carrega o índice FAISS a partir de um caminho relativo à raiz do app."""
    caminho_completo = APP_ROOT / caminho_relativo
    try:
        return faiss.read_index(str(caminho_completo))
    except Exception as e:
        st.error(f"Erro ao carregar o índice FAISS de '{caminho_completo}': {e}")
        st.stop()

@st.cache_data
def carregar_metadados(caminho_relativo: str) -> pd.DataFrame:
    """Carrega o DataFrame de metadados a partir de um caminho relativo à raiz do app."""
    caminho_completo = APP_ROOT / caminho_relativo
    try:
        with open(caminho_completo, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Arquivo de metadados não encontrado em '{caminho_completo}'. Verifique o caminho.")
        st.stop()

@st.cache_data
def carregar_devolutivas() -> pd.DataFrame:
    """Carrega o CSV com os textos das devolutivas."""
    caminho_completo = APP_ROOT / "data" / "devolutivas.csv"
    try:
        df = pd.read_csv(caminho_completo, sep=";")
        return df.rename(columns={"Necessidaes formativas": "Necessidades formativas"})
    except FileNotFoundError:
        st.error(f"Arquivo de devolutivas não encontrado em '{caminho_completo}'.")
        st.stop()

@st.cache_data
def carregar_rubricas() -> pd.DataFrame:
    """Carrega o CSV com as faixas de pontuação das rubricas."""
    caminho_completo = APP_ROOT / "data" / "rubricas.csv"
    try:
        df = pd.read_csv(caminho_completo, sep=";")
        return df
    except FileNotFoundError:
        st.error(f"Arquivo de rubricas não encontrado em '{caminho_completo}'.")
        st.stop()

# === FUNÇÕES AUXILIARES DE LÓGICA E FORMATAÇÃO ===

def exibir_cabecalho():
    """Exibe o cabeçalho padronizado da página."""
    st.title("Plataforma de Apoio à Gestão Pedagógica")
    st.markdown("Ferramenta para geração de devolutivas e recomendação de materiais.")
    st.markdown("---")

def encontrar_rubrica(df_rubricas: pd.DataFrame, pontuacao: int, dimensao: str, subdimensao: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Encontra a rubrica, nome e faixa de nível com base na pontuação do usuário."""
    candidatos = df_rubricas[
        (df_rubricas['dimensao'] == dimensao) &
        (df_rubricas['subdimensao'] == subdimensao) &
        (df_rubricas['faixa_total_min'] <= pontuacao) &
        (df_rubricas['faixa_total_max'] >= pontuacao)
    ]
    if candidatos.empty: return None, None, None
    
    rubrica_numero = int(candidatos.iloc[0]['rubrica_numero'])
    rubrica_nome = candidatos.iloc[0]['rubrica_nome']
    
    faixa = candidatos[
        (candidatos['subfaixa_min'] <= pontuacao) &
        (candidatos['subfaixa_max'] >= pontuacao)
    ]
    if faixa.empty: return rubrica_numero, rubrica_nome, None
        
    tipo_faixa = faixa.iloc[0]['tipo_faixa']
    
    return rubrica_numero, rubrica_nome, tipo_faixa

# CORREÇÃO: Esta função foi movida para ANTES da função que a chama.
def formatar_necessidades_formativas(texto: Optional[str]) -> str:
    """Formata o texto de necessidades formativas em uma lista Markdown."""
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

def gerar_texto_devolutiva_markdown(df_devolutivas: pd.DataFrame, df_rubricas: pd.DataFrame, pontuacao: int, dimensao: str, subdimensao: str) -> Optional[str]:
    """Gera o card completo da devolutiva em formato Markdown para exibição."""
    rubrica_numero, rubrica_nome, tipo_faixa = encontrar_rubrica(df_rubricas, pontuacao, dimensao, subdimensao)
    if rubrica_numero is None or tipo_faixa is None:
        st.warning(f"Não foi encontrada uma rubrica ou faixa de nível correspondente para a pontuação {pontuacao} na subdimensão '{subdimensao}'.")
        return None

    devolutiva = df_devolutivas[
        (df_devolutivas['Dimensão'] == dimensao) &
        (df_devolutivas['Subdimensão'] == subdimensao) &
        (df_devolutivas['Rubrica numero'] == rubrica_numero) &
        (df_devolutivas['Rubrica nome'] == f"{rubrica_nome} – Nível {tipo_faixa}")
    ]
    if devolutiva.empty:
        st.warning(f"O texto da devolutiva não foi encontrado para a Rubrica {rubrica_numero} - Nível {tipo_faixa}.")
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

**✅ Seus pontos fortes:**

{item['Pontos fortes']}

---

**📈 O que fazer para avançar:**

{item['O que fazer para avançar']}

---

**📚 Necessidades formativas:**

{formatar_necessidades_formativas(item['Necessidades formativas'])}
""".strip()

def gerar_texto_devolutiva_rico(df_devolutivas: pd.DataFrame, df_rubricas: pd.DataFrame, pontuacao: int, dimensao: str, subdimensao: str, modelo_selecionado: str) -> Optional[str]:
    """Gera o texto enriquecido da devolutiva para ser usado como query na busca vetorial."""
    rubrica_numero, rubrica_nome, tipo_faixa = encontrar_rubrica(df_rubricas, pontuacao, dimensao, subdimensao)
    if rubrica_numero is None or tipo_faixa is None: return None
    
    devolutiva = df_devolutivas[
        (df_devolutivas['Dimensão'] == dimensao) &
        (df_devolutivas['Subdimensão'] == subdimensao) &
        (df_devolutivas['Rubrica numero'] == rubrica_numero) &
        (df_devolutivas['Rubrica nome'] == f"{rubrica_nome} – Nível {tipo_faixa}")
    ]
    if devolutiva.empty: return None
    
    item = devolutiva.iloc[0]
    
    if modelo_selecionado == "Modelo Avançado (v2, Re-ranking)":
        contexto_query = f"Perfil do usuário: gestor no Nível {tipo_faixa} da Rubrica {rubrica_numero} - {rubrica_nome}. A necessidade de aprendizagem é a seguinte:"
        return f"{contexto_query}\n\n{item['Necessidades formativas']}".strip()
    else:
        return f"Necessidades formativas:\n{item['Necessidades formativas']}".strip()

def gerar_embedding_para_rag(modelo_st: SentenceTransformer, texto: str) -> np.ndarray:
    """Gera e normaliza um embedding para um dado texto."""
    embedding = modelo_st.encode([texto])
    return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

def gerar_card_material(row: Dict[str, Any], i: int) -> str:
    """Gera o código Markdown para exibir um card de material recomendado."""
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

def obter_pontuacao_maxima(df_rubricas: pd.DataFrame, dimensao: str, subdimensao: str) -> int:
    """Calcula a pontuação máxima para uma dada dimensão e subdimensão."""
    rubricas_filtradas = df_rubricas[
        (df_rubricas['dimensao'] == dimensao) &
        (df_rubricas['subdimensao'] == subdimensao)
    ]
    if rubricas_filtradas.empty:
        return 51
    return int(rubricas_filtradas['faixa_total_max'].max())

def sintetizar_devolutiva_com_ia(client: OpenAI, modelo_gpt: str, prompt: str, max_tokens: int) -> Optional[str]:
    """Chama a API da OpenAI para sintetizar um texto de devolutiva."""
    try:
        response = client.chat.completions.create(
            model=modelo_gpt,
            messages=[
                {"role": "system", "content": "Você é um especialista em formação de professores e gestão escolar."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao se comunicar com a API da OpenAI: {e}")
        return None