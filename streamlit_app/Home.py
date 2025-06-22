# Home.py

import streamlit as st
from src.utils import exibir_cabecalho

# --- CONFIGURAÇÃO DA PÁGINA (DEVE SER O PRIMEIRO COMANDO) ---
st.set_page_config(
    page_title="Plataforma de Apoio à Gestão",
    page_icon="📘",
    layout="wide"
)

# A barra lateral agora será gerada automaticamente pelo Streamlit
# com base nos arquivos da pasta 'pages'. Não precisamos mais configurá-la aqui.

# --- CONTEÚDO DA PÁGINA INICIAL ---
exibir_cabecalho()

st.header("Bem-vindo(a) à Plataforma!")
st.markdown(
    """
    Esta ferramenta foi projetada para apoiar gestores pedagógicos em duas frentes principais:

    1.  **Recomendação Individual:** Gere devolutivas personalizadas com base na pontuação de um profissional e receba uma lista de materiais de formação recomendados.
    
    2.  **Devolutiva Geral:** Obtenha um texto de síntese consolidado para as dimensões "Planejamento Pedagógico" ou "Pessoal-Relacional".

    **Como usar:**
    - Utilize o menu de navegação na barra lateral à esquerda para acessar as diferentes ferramentas.
    """
)