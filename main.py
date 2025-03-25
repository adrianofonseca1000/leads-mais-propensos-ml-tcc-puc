"""
Script principal para execução do pipeline completo de análise preditiva de Leads.

Este script executa todas as etapas do pipeline:
1. Coleta de dados
2. Processamento e tratamento dos dados
3. Visualização e análise exploratória
4. Treinamento e avaliação de modelos

Uso:
    python main.py [etapa]

Onde [etapa] pode ser:
    collect - Apenas coleta de dados
    process - Apenas processamento de dados
    visualize - Apenas visualização de dados
    train - Apenas treinamento de modelos
    all - Executa todas as etapas (padrão)
"""
import os
import sys
import time
from src.data.data_collection import collect_all_data
from src.features.data_processing import process_data
from src.visualization.data_visualization import create_visualizations
from src.models.model_training import train_and_evaluate

def main(step='all'):
    """
    Executa o pipeline de análise preditiva de Leads.
    
    Args:
        step: Etapa a ser executada: 'collect', 'process', 'visualize', 'train' ou 'all'.
    """
    # Marcar o tempo de início
    start_time = time.time()
    
    # Executar etapas conforme solicitado
    if step in ['collect', 'all']:
        print("\n=== Etapa 1: Coleta de dados ===")
        data = collect_all_data()
        print(f"Dados coletados: {data.shape[0]} linhas e {data.shape[1]} colunas")
    
    if step in ['process', 'all']:
        print("\n=== Etapa 2: Processamento e tratamento dos dados ===")
        input_file = 'data_raw.csv' if os.path.exists('data_raw.csv') else 'data.csv'
        data = process_data(input_file, 'data_processed.csv')
        print(f"Dados processados: {data.shape[0]} linhas e {data.shape[1]} colunas")
    
    if step in ['visualize', 'all']:
        print("\n=== Etapa 3: Visualização e análise exploratória ===")
        create_visualizations('data_processed.csv', 'visualizations')
    
    if step in ['train', 'all']:
        print("\n=== Etapa 4: Treinamento e avaliação de modelos ===")
        results = train_and_evaluate('data_processed.csv', 'models')
        
        # Mostrar resultados dos modelos
        print("\nResultados dos modelos:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
    
    # Calcular o tempo total
    elapsed_time = time.time() - start_time
    print(f"\nTempo total de execução: {elapsed_time:.2f} segundos")

if __name__ == "__main__":
    # Verificar argumentos da linha de comando
    if len(sys.argv) > 1:
        step = sys.argv[1].lower()
        valid_steps = ['collect', 'process', 'visualize', 'train', 'all']
        
        if step not in valid_steps:
            print(f"Etapa inválida: {step}")
            print(f"Escolha uma das etapas válidas: {', '.join(valid_steps)}")
            sys.exit(1)
    else:
        step = 'all'
    
    # Executar o pipeline
    main(step) 