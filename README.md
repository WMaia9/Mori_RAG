# 📘 Plataforma de Apoio à Gestão Pedagógica

Este projeto é uma aplicação web desenvolvida com Streamlit, projetada para auxiliar gestores pedagógicos na análise de desempenho e na formação continuada de professores. A ferramenta oferece devolutivas personalizadas e recomenda materiais de estudo com base em rubricas de avaliação.

## ✨ Funcionalidades Principais

* **Geração de Devolutivas Personalizadas:** Cria textos detalhados com pontos fortes, pontos a avançar e necessidades formativas com base em pontuações de avaliação.
* **Recomendação Inteligente de Materiais:** Utiliza busca vetorial (Sentence Transformers + FAISS) e um sistema de re-ranking ponderado para sugerir os materiais mais relevantes da base de dados.
* **Comparação de Modelos de IA:** Permite ao usuário alternar em tempo real entre diferentes motores de recomendação (Legado, Busca Simples e Avançado com Re-ranking) para comparar a qualidade e a filosofia de cada abordagem.
* **Síntese de Devolutivas Gerais:** Usa um modelo de linguagem generativo (OpenAI GPT) para criar um texto consolidado para equipes ou para a escola como um todo.

## 📂 Estrutura do Projeto

O projeto é organizado com a seguinte estrutura de pastas e arquivos para garantir a separação de responsabilidades e facilitar a manutenção:

-   `PROJETO_MORI/` (Pasta Raiz do Projeto)
    -   `.gitignore`
    -   `README.md`
    -   `requirements.txt`
    -   **`data_source/`**: Contém os dados brutos e originais para o processamento.
        -   `Base_de_ODAS_1606.xlsx`
    -   **`notebooks/`**: Contém os Jupyter Notebooks para exploração e geração dos modelos.
        -   `1_Geracao_Embeddings.ipynb`
    -   **`streamlit_app/`**: A pasta principal da aplicação web.
        -   `.streamlit/`
            -   `config.toml`
        -   `assets/`
            -   *(logos e imagens)*
        -   `data/`
            -   `devolutivas.csv`
            -   `rubricas.csv`
        -   `models/`
            -   `odas_index_1606_v2.faiss`
            -   `metadados_odas_1606_v2.pkl`
            -   *(e outras versões...)*
        -   `pages/`
            -   `1_Recomendacao_Individual.py`
            -   `2_Devolutiva_Geral.py`
        -   `src/`
            -   `__init__.py`
            -   `recommendation.py`
            -   `utils.py`
        -   `Home.py`

## 🛠️ Tecnologias Utilizadas

* **Interface Web:** Streamlit
* **Manipulação de Dados:** Pandas, NumPy
* **IA & Busca Semântica:** Sentence-Transformers, FAISS
* **Síntese de Texto:** OpenAI API

## 🚀 Instalação e Execução

Siga os passos abaixo para configurar e rodar o projeto localmente.

### 1. Pré-requisitos

-   Python 3.9 ou superior
-   Git
-   **Git LFS** (para lidar com arquivos de modelo grandes). Instale a partir de [git-lfs.github.com](https://git-lfs.github.com/).

### 2. Setup do Ambiente

```bash
# 1. Clone o repositório para a sua máquina
git clone [URL_DO_SEU_REPOSITORIO]
cd PROJETO_MORI

# 2. Ative o Git LFS (só precisa fazer uma vez por repositório)
git lfs install
git lfs pull

# 3. Crie e ative um ambiente virtual
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# 4. Instale todas as dependências
pip install -r requirements.txt