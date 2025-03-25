"""
Módulo para processamento e tratamento dos dados.

Este módulo contém funções para limpeza, transformação e preparação dos dados
para a fase de modelagem.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_path='data_raw.csv'):
    """
    Carrega os dados do arquivo CSV.
    
    Args:
        file_path: Caminho para o arquivo CSV.
        
    Returns:
        DataFrame: Dados carregados.
    """
    data = pd.read_csv(file_path, sep=';')
    print(f'O dataset tem {data.shape[0]} linhas e {data.shape[1]} colunas')
    return data

def handle_missing_values(data):
    """
    Trata valores ausentes nos dados.
    
    Args:
        data: DataFrame com os dados.
        
    Returns:
        DataFrame: Dados com valores ausentes tratados.
    """
    # Verificando valores ausentes
    missing_data = data.isnull().sum()
    print("Valores ausentes por coluna:")
    print(missing_data[missing_data > 0])
    
    # Para colunas numéricas, substituir NaN por 0
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        data[col] = data[col].fillna(0)
    
    return data

def encode_categorical_variables(data):
    """
    Codifica variáveis categóricas usando LabelEncoder.
    
    Args:
        data: DataFrame com os dados.
        
    Returns:
        DataFrame: Dados com variáveis categóricas codificadas.
        dict: Dicionário com os codificadores para cada coluna.
    """
    # Identificar colunas categóricas
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Criar um dicionário para armazenar os codificadores
    encoders = {}
    
    # Aplicar LabelEncoder para cada coluna categórica
    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder
    
    return data, encoders

def filter_and_balance_data(data, target_col='venda', balance_ratio=None):
    """
    Filtra e balanceia os dados para a modelagem.
    
    Args:
        data: DataFrame com os dados.
        target_col: Nome da coluna alvo.
        balance_ratio: Proporção desejada entre classes (opcional).
        
    Returns:
        DataFrame: Dados filtrados e balanceados.
    """
    # Verificar a distribuição da variável alvo
    target_counts = data[target_col].value_counts()
    print("Distribuição da variável alvo:")
    print(target_counts)
    
    # Se balance_ratio for especificado, balancear os dados
    if balance_ratio:
        min_class = target_counts.idxmin()
        max_class = target_counts.idxmax()
        min_count = target_counts[min_class]
        max_count = target_counts[max_class]
        
        # Calcular quantas amostras da classe majoritária manter
        samples_to_keep = int(min_count * balance_ratio)
        
        # Separar as classes
        df_min = data[data[target_col] == min_class]
        df_max = data[data[target_col] == max_class].sample(samples_to_keep, random_state=42)
        
        # Combinar os dados balanceados
        data_balanced = pd.concat([df_min, df_max])
        
        # Embaralhar os dados
        data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Dados balanceados: {data_balanced.shape[0]} linhas")
        return data_balanced
    
    return data

def remove_outliers(data, cols=None, threshold=3):
    """
    Remove outliers utilizando o método do Z-Score.
    
    Args:
        data: DataFrame com os dados.
        cols: Lista de colunas para verificar outliers. Se None, usa todas as colunas numéricas.
        threshold: Limiar do Z-Score para considerar um valor como outlier.
        
    Returns:
        DataFrame: Dados sem outliers.
    """
    if cols is None:
        cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    data_no_outliers = data.copy()
    rows_before = data_no_outliers.shape[0]
    
    # Para cada coluna, calcular o Z-Score e remover outliers
    for col in cols:
        z_scores = np.abs((data_no_outliers[col] - data_no_outliers[col].mean()) / data_no_outliers[col].std())
        data_no_outliers = data_no_outliers[z_scores < threshold]
    
    rows_after = data_no_outliers.shape[0]
    print(f"Removidos {rows_before - rows_after} outliers.")
    
    return data_no_outliers

def normalize_features(data, target_col='venda'):
    """
    Normaliza as features para ter média 0 e desvio padrão 1.
    
    Args:
        data: DataFrame com os dados.
        target_col: Nome da coluna alvo.
        
    Returns:
        DataFrame: Dados com features normalizadas.
    """
    # Separar a variável alvo
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Normalizar as features
    X_normalized = (X - X.mean()) / X.std()
    
    # Recombinar com a variável alvo
    data_normalized = pd.concat([X_normalized, y], axis=1)
    
    return data_normalized

def process_data(input_file='data_raw.csv', output_file='data_processed.csv'):
    """
    Executa todo o fluxo de processamento de dados.
    
    Args:
        input_file: Caminho para o arquivo de entrada.
        output_file: Caminho para o arquivo de saída.
        
    Returns:
        DataFrame: Dados processados.
    """
    # Carregar os dados
    data = load_data(input_file)
    
    # Tratar valores ausentes
    data = handle_missing_values(data)
    
    # Codificar variáveis categóricas
    data, encoders = encode_categorical_variables(data)
    
    # Remover outliers
    data = remove_outliers(data)
    
    # Balancear os dados (1:1)
    data = filter_and_balance_data(data, balance_ratio=1.0)
    
    # Salvar os dados processados
    data.to_csv(output_file, sep=';', index=False)
    
    print(f"Dados processados salvos em {output_file}")
    return data

if __name__ == "__main__":
    # Verificar se o arquivo de dados brutos existe
    if not os.path.exists('data_raw.csv'):
        print("Arquivo data_raw.csv não encontrado. Execute o módulo de coleta de dados primeiro.")
    else:
        # Executar o processamento de dados
        process_data() 