# 📘 Plataforma de Apoio à Gestão Pedagógica

Este projeto é uma aplicação web desenvolvida com Streamlit, projetada para auxiliar gestores pedagógicos na análise de desempenho e na formação continuada de professores. A ferramenta oferece devolutivas personalizadas e recomenda materiais de estudo com base em rubricas de avaliação.

## ✨ Funcionalidades Principais

* **Geração de Devolutivas Personalizadas:** Cria textos detalhados com pontos fortes, pontos a avançar e necessidades formativas com base em pontuações de avaliação.
* **Recomendação Inteligente de Materiais:** Utiliza busca vetorial (Sentence Transformers + FAISS) e um sistema de re-ranking ponderado para sugerir os materiais mais relevantes da base de dados.
* **Comparação de Modelos de IA:** Permite ao usuário alternar em tempo real entre diferentes motores de recomendação (Legado, Busca Simples e Avançado com Re-ranking) para comparar a qualidade e a filosofia de cada abordagem.
* **Síntese de Devolutivas Gerais:** Usa um modelo de linguagem generativo (OpenAI GPT) para criar um texto consolidado para equipes ou para a escola como um todo.

## 📂 Estrutura do Projeto

O projeto é organizado com uma estrutura modular para separar a preparação dos dados da aplicação final, facilitando a manutenção e a escalabilidade. 📂 PROJETO_MORI/
│
├── 📜 .gitignore
├── 📜 README.md
└── 📜 requirements.txt
│
├── 📂 notebooks/
│   └── 📄 1_Geracao_Embeddings.ipynb
│
├── 📂 data_source/
│   └── 📄 Base_de_ODAS_1606.xlsx
│
└── 📂 streamlit_app/
│
├── 📂 .streamlit/ & 📂 assets/
├── 📂 data/
│   └── 📄 devolutivas.csv, rubricas.csv
├── 📂 models/
│   └── 📄 *.faiss, *.pkl
├── 📂 pages/
│   ├── 🐍 1_Recomendacao_Individual.py
│   └── 🐍 2_Devolutiva_Geral.py
│
├── 📂 src/
│   ├── 📄 init.py
│   ├── 🐍 recommendation.py
│   └── 🐍 utils.py
│
└── 🐍 Home.py


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