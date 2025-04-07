# Análise Preditiva de Leads - TCC PUC

## Descrição

Este projeto foi desenvolvido como Trabalho de Conclusão de Curso da Pós-Graduação em Data Science and Big Data da PUC. 
O objetivo é identificar e atribuir um score de probabilidade para Leads receptivos à compra de planos de internet, 
utilizando técnicas de Data Science e Machine Learning.

Com a solução, os times de vendas, planejamento e marketing podem priorizar os Leads com maior interesse nos planos de internet.

## Estrutura do Projeto

O projeto foi organizado de forma modular em uma estrutura de pacotes Python:

```
leads-mais-propensos-ml-tcc-puc/
│
├── data/
│   ├── raw/             # Original raw data (e.g., data.csv)
│   └── processed/       # Processed data (e.g., data_ptd.csv, data_aed.csv)
│
├── notebooks/           # Original Jupyter notebooks for reference
│   ├── 01 - Coleta de dados.ipynb
│   ├── 02 - Processamento e tratamento dos dados.ipynb
│   ├── 03 - Análise e exploração dos dados.ipynb
│   └── 04 - Aplicação e avaliação de Modelos Machine Learning.ipynb
│
├── reports/             # Generated reports and figures
│   └── figures/         # Plots and visualizations (e.g., descritiva*.html)
│
├── models/              # Trained models outputs
│
├── src/                 # Source code for the project
│   ├── __init__.py
│   ├── data_processing.py  # Code from notebook 02
│   ├── exploratory_analysis.py # Code from notebook 03
│   ├── modeling.py         # Code from notebook 04
│   ├── predict.py          # Script for making predictions with a trained model
│   ├── train.py            # Script for training the model
│   └── utils.py            # Utility functions (shared code)
│
├── .gitignore           # Specifies intentionally untracked files that Git should ignore
├── requirements.txt     # Project dependencies
└── README.md            # Updated project documentation
```

## Instalação

1. Clone o repositório:
```
git clone <url-do-repositorio>
cd <diretorio-do-projeto>
```

2. Crie e ative um ambiente virtual:
```
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Instale as dependências:
```
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```
cp .env.example .env
```
Edite o arquivo `.env` com suas credenciais de banco de dados.

## Uso

Para executar o pipeline completo:
```
python main.py
```

Para executar etapas específicas:
```
python main.py collect    # Apenas coleta de dados
python main.py process    # Apenas processamento de dados
python main.py visualize  # Apenas visualização de dados
python main.py train      # Apenas treinamento de modelos
```

Para fazer predições com novos dados:
```
python -m src.models.prediction <arquivo_entrada> <arquivo_saida>
```

## Pipeline de Análise

O projeto segue um pipeline de análise preditiva com as seguintes etapas:

1. **Coleta de Dados**: Conexão ao banco de dados e extração de informações de contato, recargas e serviços dos Leads.

2. **Processamento e Tratamento dos Dados**: Limpeza, transformação e preparação dos dados para modelagem.

3. **Visualização e Análise Exploratória**: Geração de visualizações para compreensão dos padrões nos dados.

4. **Treinamento e Avaliação de Modelos**: Treinamento de diferentes modelos de classificação (Random Forest, SVM, Naive Bayes) e avaliação de desempenho.

## Tecnologias Utilizadas

- Coleta de Dados: Python, SQL, pyodbc
- Processamento e Tratamento: pandas, numpy, scikit-learn
- Visualização: matplotlib, seaborn
- Modelagem: scikit-learn

## Dados Analisados

Este estudo considera Leads espalhados pelo território brasileiro durante o terceiro e quarto trimestre do ano 2021.
