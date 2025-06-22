# 📘 Plataforma de Apoio à Gestão Pedagógica

Este projeto é uma aplicação web desenvolvida com Streamlit, projetada para auxiliar gestores pedagógicos na análise de desempenho e na formação continuada de professores. A ferramenta oferece devolutivas personalizadas e recomenda materiais de estudo com base em rubricas de avaliação.

## ✨ Funcionalidades Principais

* **Devolutiva Individual:** Gera um feedback detalhado para um profissional com base em sua pontuação em diferentes dimensões e subdimensões pedagógicas.
* **Recomendação Inteligente de Materiais:** Utiliza busca vetorial (Sentence Transformers + FAISS) para encontrar e recomendar materiais de formação (textos, vídeos, áudios, etc.) relevantes para as necessidades identificadas na devolutiva.
* **Motores de Recomendação Comparáveis:** Permite ao usuário alternar entre diferentes modelos de recomendação para comparar os resultados:
    * **Modelo Simples:** Busca por similaridade semântica em toda a base de dados.
    * **Modelo Avançado:** Utiliza uma arquitetura de Filtro Inteligente + Re-ranking Ponderado, alavancando metadados gerados por IA para uma precisão superior.
* **Síntese de Devolutiva Geral:** Utiliza um modelo de linguagem generativo (GPT) para criar um texto consolidado para equipes ou para a escola como um todo.

## 📂 Estrutura do Projeto

O projeto é organizado da seguinte forma para separar a preparação dos dados da aplicação final:

-   `notebooks/`: Contém os Jupyter Notebooks para processamento de dados e geração dos embeddings e modelos.
-   `data_source/`: Local para os dados brutos (ex: planilhas Excel) que alimentam os notebooks.
-   `streamlit_app/`: Contém o código-fonte completo da aplicação web.
    -   `models/`: Armazena os artefatos gerados (índices `.faiss` e metadados `.pkl`).
    -   `pages/`: Contém as diferentes páginas da aplicação.
-   `requirements.txt`: Lista todas as dependências do projeto.

## 🚀 Instalação e Execução

Siga os passos abaixo para rodar o projeto localmente.

### 1. Pré-requisitos

- Python 3.9 ou superior
- Git

### 2. Setup do Ambiente

```bash
# 1. Clone o repositório para a sua máquina
git clone [URL_DO_SEU_REPOSITORIO]
cd PROJETO_MORI

# 2. Crie e ative um ambiente virtual
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# 3. Instale todas as dependências
pip install -r requirements.txt