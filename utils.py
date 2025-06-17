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
from typing import Tuple, List, Dict, Any, Optional
from openai import OpenAI
from typing import Optional

# === FUNÇÕES DE CACHE PARA CARREGAMENTO DE DADOS ===

@st.cache_resource
def carregar_modelo_st() -> SentenceTransformer:
    """Carrega o modelo de embedding SentenceTransformer e o mantém em cache.

    Returns:
        SentenceTransformer: O objeto do modelo carregado.
    """
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

@st.cache_resource
def carregar_index(caminho: str) -> faiss.Index:
    """Carrega o índice FAISS do disco e o mantém em cache.

    Args:
        caminho (str): O caminho para o arquivo .faiss.

    Returns:
        faiss.Index: O objeto do índice FAISS carregado.
    """
    return faiss.read_index(caminho)

@st.cache_data
def carregar_metadados(caminho: str) -> pd.DataFrame:
    """Carrega o DataFrame de metadados do disco e o mantém em cache.

    Args:
        caminho (str): O caminho para o arquivo .pkl.

    Returns:
        pd.DataFrame: O DataFrame com os metadados dos materiais.
    """
    with open(caminho, "rb") as f:
        return pickle.load(f)

@st.cache_data
def carregar_devolutivas() -> pd.DataFrame:
    """Carrega e prepara o CSV com os textos das devolutivas."""
    df = pd.read_csv("data/devolutivas.csv", sep=";")
    return df.rename(columns={"Necessidaes formativas": "Necessidades formativas"})

@st.cache_data
def carregar_rubricas() -> pd.DataFrame:
    """Carrega e prepara o CSV com as faixas de pontuação das rubricas."""
    return pd.read_csv("data/rubricas.csv", sep=";")


# === FUNÇÕES AUXILIARES DE LÓGICA E FORMATAÇÃO ===

def encontrar_rubrica(df_rubricas: pd.DataFrame, pontuacao: int, dimensao: str, subdimensao: str) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    """Encontra a rubrica, nome e faixa de nível com base na pontuação do usuário.

    Args:
        df_rubricas (pd.DataFrame): DataFrame contendo as regras das rubricas.
        pontuacao (int): A pontuação do usuário.
        dimensao (str): A dimensão avaliada.
        subdimensao (str): A subdimensão avaliada.

    Returns:
        Tuple[Optional[int], Optional[str], Optional[int]]: Uma tupla contendo o número da rubrica,
        o nome da rubrica e o tipo da faixa. Retorna (None, None, None) se não encontrar.
    """
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


def formatar_necessidades_formativas(texto: Optional[str]) -> str:
    """Formata o texto de necessidades formativas em uma lista Markdown.

    Args:
        texto (Optional[str]): O texto bruto vindo do CSV.

    Returns:
        str: O texto formatado como uma lista de itens em Markdown.
    """
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
    """Gera o card completo da devolutiva em formato Markdown para exibição.

    Args:
        df_devolutivas (pd.DataFrame): DataFrame com os textos das devolutivas.
        df_rubricas (pd.DataFrame): DataFrame com as regras das rubricas.
        pontuacao (int): Pontuação do usuário.
        dimensao (str): Dimensão avaliada.
        subdimensao (str): Subdimensão avaliada.

    Returns:
        Optional[str]: Uma string formatada em Markdown com a devolutiva completa,
                       ou None se não for encontrada.
    """
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
    """Gera o texto enriquecido da devolutiva para ser usado como query na busca vetorial.

    Args:
        df_devolutivas (pd.DataFrame): DataFrame com os textos das devolutivas.
        df_rubricas (pd.DataFrame): DataFrame com as regras das rubricas.
        pontuacao (int): Pontuação do usuário.
        dimensao (str): Dimensão avaliada.
        subdimensao (str): Subdimensão avaliada.
        modelo_selecionado (str): Nome do modelo de busca ativo, para decidir se enriquece a query.

    Returns:
        Optional[str]: O texto-query para a busca, ou None se não for encontrada.
    """
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
    
    # A query só é enriquecida para o modelo avançado, que foi treinado para entender esse contexto
    if modelo_selecionado == "Modelo Avançado (v2, Re-ranking)":
        contexto_query = f"Perfil do usuário: gestor no Nível {tipo_faixa} da Rubrica {rubrica_numero} - {rubrica_nome}. A necessidade de aprendizagem é a seguinte:"
        return f"{contexto_query}\n\n{item['Necessidades formativas']}".strip()
    else:
        # Para os outros modelos, uma query mais simples é mais eficaz
        return f"Necessidades formativas:\n{item['Necessidades formativas']}".strip()


def gerar_embedding_para_rag(modelo_st: SentenceTransformer, texto: str) -> np.ndarray:
    """Gera e normaliza um embedding para um dado texto.

    Args:
        modelo_st (SentenceTransformer): O modelo de embedding carregado.
        texto (str): O texto a ser vetorizado.

    Returns:
        np.ndarray: O vetor de embedding normalizado.
    """
    embedding = modelo_st.encode([texto])
    return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)


def gerar_card_material(row: Dict[str, Any], i: int) -> str:
    """Gera o código Markdown para exibir um card de material recomendado.

    Args:
        row (Dict[str, Any]): Um dicionário contendo os dados de uma linha do DataFrame de resultados.
        i (int): O índice do material na lista (para a numeração).

    Returns:
        str: A string Markdown formatada para o card.
    """
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
    """Calcula a pontuação máxima para uma dada dimensão e subdimensão.

    Args:
        df_rubricas (pd.DataFrame): DataFrame com as regras das rubricas.
        dimensao (str): A dimensão avaliada.
        subdimensao (str): A subdimensão avaliada.

    Returns:
        int: O valor da pontuação máxima, ou 51 como padrão se não for encontrado.
    """
    rubricas_filtradas = df_rubricas[
        (df_rubricas['dimensao'] == dimensao) &
        (df_rubricas['subdimensao'] == subdimensao)
    ]
    if rubricas_filtradas.empty:
        return 51
    return int(rubricas_filtradas['faixa_total_max'].max())

def sintetizar_devolutiva_com_ia(client: OpenAI, modelo_gpt: str, prompt: str, max_tokens: int) -> Optional[str]:
    """
    Chama a API da OpenAI para sintetizar um texto de devolutiva.

    Args:
        client (OpenAI): O cliente da API OpenAI instanciado.
        modelo_gpt (str): O nome do modelo GPT a ser usado (ex: 'gpt-4o-mini').
        prompt (str): O prompt completo a ser enviado para a IA.
        max_tokens (int): O número máximo de tokens para a resposta.

    Returns:
        Optional[str]: O texto de resposta da IA, ou None em caso de erro.
    """
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