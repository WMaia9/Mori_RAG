# pages/2_Devolutiva_Geral.py

import streamlit as st
from openai import OpenAI
from utils import * # Importa todas as funções de utils.py

# --- CABEÇALHO E CARREGAMENTO DE DADOS ---
exibir_cabecalho()

df_devolutivas = carregar_devolutivas()
df_rubricas = carregar_rubricas()

# Lê a escolha do modelo do st.session_state para passar para a função de geração de texto
modelo_ativo = st.session_state.get('modelo_ativo', "Modelo Avançado (v2, Re-ranking)")

# --- INTERFACE DA PÁGINA ---
st.header("Devolutiva Geral da Dimensão")

st.sidebar.markdown("### 🤖 Configurações de IA (Síntese)")
modelo_gpt_selecionado = st.sidebar.selectbox(
    "Escolha o modelo de IA:", ["gpt-4o-mini", "gpt-4"], index=0,
    help="Usado para gerar o texto de síntese nesta página."
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
                try:
                    response = client.chat.completions.create(
                        model=modelo_gpt_selecionado,
                        messages=[{"role": "system", "content": "Você é um especialista em formação de professores."}, {"role": "user", "content": prompt}],
                        temperature=0.7, max_tokens=1500
                    )
                    st.markdown("### 📖 Devolutiva da Dimensão Pedagógica")
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Erro ao gerar devolutiva: {str(e)}")


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
                try:
                    response = client.chat.completions.create(
                        model=modelo_gpt_selecionado,
                        messages=[{"role": "system", "content": "Você é um especialista em formação de professores."}, {"role": "user", "content": prompt}],
                        temperature=0.7, max_tokens=1000
                    )
                    st.markdown("### 📖 Devolutiva da Dimensão Pessoal-Relacional")
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Erro ao gerar devolutiva: {str(e)}")