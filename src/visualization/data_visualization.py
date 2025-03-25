"""
Módulo para visualização e análise exploratória dos dados.

Este módulo contém funções para gerar visualizações e análises dos dados,
auxiliando na compreensão do problema e na identificação de padrões.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

def plot_class_distribution(data, target_col='venda', figsize=(10, 6)):
    """
    Plota a distribuição da variável alvo.
    
    Args:
        data: DataFrame com os dados.
        target_col: Nome da coluna alvo.
        figsize: Tamanho da figura.
    """
    plt.figure(figsize=figsize)
    sns.countplot(x=target_col, data=data)
    plt.title('Distribuição da variável alvo')
    plt.xlabel('Classe')
    plt.ylabel('Contagem')
    plt.savefig('target_distribution.png')
    plt.close()

def plot_correlation_matrix(data, figsize=(12, 10)):
    """
    Plota a matriz de correlação das variáveis.
    
    Args:
        data: DataFrame com os dados.
        figsize: Tamanho da figura.
    """
    # Calcular a matriz de correlação
    corr_matrix = data.corr()
    
    # Criar o heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, annot=False, fmt='.2f')
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def plot_feature_importance(data, target_col='venda', top_n=20, figsize=(12, 10)):
    """
    Plota a importância das features baseada na correlação com a variável alvo.
    
    Args:
        data: DataFrame com os dados.
        target_col: Nome da coluna alvo.
        top_n: Número de features a serem exibidas.
        figsize: Tamanho da figura.
    """
    # Calcular correlação com a variável alvo
    target_corr = data.corr()[target_col].sort_values(ascending=False)
    
    # Selecionar as top_n features mais correlacionadas (positiva e negativamente)
    top_positive = target_corr.head(top_n + 1)  # +1 porque incluirá a própria variável alvo
    top_negative = target_corr.tail(top_n)
    
    # Remover a variável alvo da lista de features positivas
    top_positive = top_positive.drop(target_col)
    
    # Concatenar as features mais importantes
    top_features = pd.concat([top_positive, top_negative])
    
    # Criar o gráfico
    plt.figure(figsize=figsize)
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title(f'Top {top_n} Features mais correlacionadas com {target_col}')
    plt.xlabel('Correlação')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_numeric_distributions(data, n_cols=3, figsize=(18, 15)):
    """
    Plota a distribuição das variáveis numéricas.
    
    Args:
        data: DataFrame com os dados.
        n_cols: Número de colunas no grid de subplots.
        figsize: Tamanho da figura.
    """
    # Selecionar colunas numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Remover colunas com um único valor
    numeric_cols = [col for col in numeric_cols if data[col].nunique() > 1]
    
    # Calcular número de linhas necessárias
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    
    # Criar subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plotar histogramas para cada coluna numérica
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            sns.histplot(data[col], ax=axes[i], kde=True)
            axes[i].set_title(f'Distribuição de {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequência')
    
    # Remover subplots vazios
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('numeric_distributions.png')
    plt.close()

def plot_categorical_distributions(data, n_cols=3, figsize=(18, 15)):
    """
    Plota a distribuição das variáveis categóricas.
    
    Args:
        data: DataFrame com os dados.
        n_cols: Número de colunas no grid de subplots.
        figsize: Tamanho da figura.
    """
    # Selecionar colunas categóricas
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Verificar se existem colunas categóricas
    if len(categorical_cols) == 0:
        print("Não foram encontradas colunas categóricas nos dados.")
        return
    
    # Calcular número de linhas necessárias
    n_rows = int(np.ceil(len(categorical_cols) / n_cols))
    
    # Criar subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Garantir que axes seja um array 2D mesmo com uma linha
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plotar countplots para cada coluna categórica
    for i, col in enumerate(categorical_cols):
        row, col_idx = divmod(i, n_cols)
        if row < n_rows and col_idx < n_cols:
            top_categories = data[col].value_counts().head(10).index
            sns.countplot(y=col, data=data[data[col].isin(top_categories)], ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Top 10 categorias de {col}')
            axes[row, col_idx].set_xlabel('Contagem')
            axes[row, col_idx].set_ylabel(col)
    
    # Remover subplots vazios
    for i in range(len(categorical_cols), n_rows * n_cols):
        row, col_idx = divmod(i, n_cols)
        if row < axes.shape[0] and col_idx < axes.shape[1]:
            fig.delaxes(axes[row, col_idx])
    
    plt.tight_layout()
    plt.savefig('categorical_distributions.png')
    plt.close()

def plot_pairplot(data, target_col='venda', features=None, figsize=(15, 15)):
    """
    Plota um pairplot para visualizar relações entre pares de variáveis.
    
    Args:
        data: DataFrame com os dados.
        target_col: Nome da coluna alvo.
        features: Lista de features a incluir no pairplot. Se None, seleciona automaticamente.
        figsize: Tamanho da figura.
    """
    # Se features não for especificado, selecionar as mais correlacionadas com o target
    if features is None:
        # Calcular correlação com a variável alvo
        target_corr = data.corr()[target_col].abs().sort_values(ascending=False)
        
        # Selecionar as 5 features mais correlacionadas
        features = target_corr.head(6).index.tolist()
        
        # Remover a variável alvo da lista de features
        features.remove(target_col)
    
    # Adicionar a variável alvo à lista de features
    features.append(target_col)
    
    # Criar o pairplot
    plt.figure(figsize=figsize)
    g = sns.pairplot(data[features], hue=target_col, diag_kind='kde')
    g.fig.suptitle('Pairplot das Features mais importantes', y=1.02)
    plt.tight_layout()
    plt.savefig('pairplot.png')
    plt.close()

def create_visualizations(data_file='data_processed.csv', output_dir='visualizations'):
    """
    Cria todas as visualizações para análise exploratória.
    
    Args:
        data_file: Caminho para o arquivo de dados processados.
        output_dir: Diretório para salvar as visualizações.
    """
    # Criar diretório de saída se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Mudar para o diretório de saída
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        # Carregar dados
        data = pd.read_csv(data_file, sep=';')
        print(f"Dados carregados: {data.shape[0]} linhas e {data.shape[1]} colunas")
        
        # Gerar visualizações
        plot_class_distribution(data)
        plot_correlation_matrix(data)
        plot_feature_importance(data)
        plot_numeric_distributions(data)
        plot_categorical_distributions(data)
        plot_pairplot(data)
        
        print(f"Visualizações salvas em {output_dir}")
    
    finally:
        # Voltar ao diretório original
        os.chdir(original_dir)

if __name__ == "__main__":
    # Verificar se o arquivo de dados processados existe
    if not os.path.exists('data_processed.csv'):
        print("Arquivo data_processed.csv não encontrado. Execute o módulo de processamento de dados primeiro.")
    else:
        # Criar as visualizações
        create_visualizations() 