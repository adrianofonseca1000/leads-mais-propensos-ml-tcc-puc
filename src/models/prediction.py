"""
Módulo para classificação de novos Leads usando o modelo treinado.

Este módulo contém funções para carregar o modelo treinado e fazer predições
para novos dados de Leads.
"""
import pandas as pd
import numpy as np
import os
import pickle
import joblib

def load_model(model_dir='models'):
    """
    Carrega o modelo, o scaler e os nomes das features.
    
    Args:
        model_dir: Diretório onde o modelo está salvo.
        
    Returns:
        model: Modelo carregado.
        scaler: Scaler carregado.
        feature_names: Nomes das features.
    """
    # Verificar se o diretório existe
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Diretório {model_dir} não encontrado.")
    
    # Carregar o modelo
    model_file = os.path.join(model_dir, 'model.pkl')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Arquivo de modelo {model_file} não encontrado.")
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Carregar o scaler
    scaler_file = os.path.join(model_dir, 'scaler.pkl')
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Arquivo de scaler {scaler_file} não encontrado.")
    
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    # Carregar os nomes das features
    feature_file = os.path.join(model_dir, 'feature_names.pkl')
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Arquivo de features {feature_file} não encontrado.")
    
    with open(feature_file, 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, scaler, feature_names

def prepare_input_data(data, feature_names, scaler):
    """
    Prepara os dados de entrada para predição.
    
    Args:
        data: DataFrame ou dict com os dados de entrada.
        feature_names: Lista de nomes das features esperadas pelo modelo.
        scaler: Scaler para normalizar os dados.
        
    Returns:
        X: Array 2D com os dados preparados para predição.
    """
    # Converter para DataFrame se for um dicionário
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Verificar se todas as features estão presentes
    missing_features = set(feature_names) - set(data.columns)
    if missing_features:
        # Preencher features ausentes com 0
        for feature in missing_features:
            data[feature] = 0
    
    # Selecionar apenas as features usadas pelo modelo e na ordem correta
    X = data[feature_names].values
    
    # Aplicar o scaler
    X_scaled = scaler.transform(X)
    
    return X_scaled

def predict(data, model_dir='models'):
    """
    Faz predições para novos dados.
    
    Args:
        data: DataFrame ou dict com os dados de entrada.
        model_dir: Diretório onde o modelo está salvo.
        
    Returns:
        predictions: Lista de predições (0 ou 1).
        probabilities: Lista de probabilidades da classe positiva.
    """
    # Carregar modelo, scaler e nomes das features
    model, scaler, feature_names = load_model(model_dir)
    
    # Preparar dados de entrada
    X = prepare_input_data(data, feature_names, scaler)
    
    # Fazer predições
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    return predictions, probabilities

def batch_predict(input_file, output_file='predictions.csv', model_dir='models'):
    """
    Faz predições em lote para um arquivo CSV.
    
    Args:
        input_file: Caminho para o arquivo CSV com os dados de entrada.
        output_file: Caminho para o arquivo CSV de saída.
        model_dir: Diretório onde o modelo está salvo.
        
    Returns:
        DataFrame com os dados de entrada e as predições.
    """
    # Carregar dados
    data = pd.read_csv(input_file, sep=';')
    print(f"Dados carregados: {data.shape[0]} linhas e {data.shape[1]} colunas")
    
    # Fazer predições
    predictions, probabilities = predict(data, model_dir)
    
    # Adicionar predições ao DataFrame
    data['predicted_class'] = predictions
    data['probability'] = probabilities
    
    # Salvar resultados
    data.to_csv(output_file, sep=';', index=False)
    print(f"Predições salvas em {output_file}")
    
    return data

def classify_lead(lead_data, threshold=0.5, model_dir='models'):
    """
    Classifica um único Lead e retorna a classe e a probabilidade.
    
    Args:
        lead_data: Dict ou DataFrame com os dados do Lead.
        threshold: Limiar para classificação.
        model_dir: Diretório onde o modelo está salvo.
        
    Returns:
        dict: Dicionário com a classe predita e probabilidade.
    """
    # Fazer predição
    _, probabilities = predict(lead_data, model_dir)
    probability = probabilities[0]
    
    # Aplicar threshold
    prediction = 1 if probability >= threshold else 0
    
    # Definir rótulo da classe
    class_label = "Propenso à compra" if prediction == 1 else "Não propenso à compra"
    
    return {
        'prediction': prediction,
        'probability': probability,
        'class': class_label
    }

if __name__ == "__main__":
    # Exemplo de uso
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"Arquivo {input_file} não encontrado.")
            sys.exit(1)
        
        # Definir arquivo de saída
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
        
        # Fazer predições em lote
        batch_predict(input_file, output_file)
    else:
        print("Uso: python prediction.py [arquivo_entrada] [arquivo_saida]")
        print("Exemplo: python prediction.py novos_leads.csv resultados.csv") 