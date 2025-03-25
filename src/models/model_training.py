"""
Módulo para treinamento e avaliação de modelos de machine learning.

Este módulo contém funções para treinar, avaliar e otimizar modelos de
classificação para predição de Leads propensos a compra.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_score, 
    accuracy_score, 
    recall_score, 
    f1_score, 
    precision_recall_curve, 
    roc_curve, 
    roc_auc_score
)

def load_data(file_path='data_processed.csv'):
    """
    Carrega os dados processados do arquivo CSV.
    
    Args:
        file_path: Caminho para o arquivo CSV.
        
    Returns:
        X: Features.
        y: Variável alvo.
        feature_names: Nomes das features.
    """
    data = pd.read_csv(file_path, sep=';')
    print(f'O dataset tem {data.shape[0]} linhas e {data.shape[1]} colunas')
    
    # Definir variáveis X e y
    y = np.array(data.venda.tolist())
    X_df = data.drop('venda', axis=1)
    feature_names = X_df.columns.tolist()
    X = np.array(X_df.to_numpy())
    
    return X, y, feature_names

def normalize_features(X_train, X_test):
    """
    Normaliza as features usando MinMaxScaler.
    
    Args:
        X_train: Dados de treino.
        X_test: Dados de teste.
        
    Returns:
        X_train_scaled: Dados de treino normalizados.
        X_test_scaled: Dados de teste normalizados.
        scaler: Objeto scaler treinado.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def plot_feature_importance(model, feature_names, output_dir='.'):
    """
    Plota a importância das features do modelo.
    
    Args:
        model: Modelo treinado com atributo feature_importances_.
        feature_names: Nomes das features.
        output_dir: Diretório para salvar a visualização.
    """
    # Verificar se o modelo tem o atributo feature_importances_
    if not hasattr(model, 'feature_importances_'):
        print("O modelo não suporta visualização de importância de features.")
        return
    
    # Criar DataFrame com a importância das features
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Criar o gráfico
    plt.figure(figsize=(10, 18))
    plt.title('Importância das Features', fontsize=20)
    sns.barplot(y='Feature', x='Importance', data=importance_df)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def plot_roc_curve(fpr, tpr, auc, output_dir='.'):
    """
    Plota a curva ROC.
    
    Args:
        fpr: Taxa de falsos positivos.
        tpr: Taxa de verdadeiros positivos.
        auc: Área sob a curva.
        output_dir: Diretório para salvar a visualização.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', label=f'ROC (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC (Receiver Operating Characteristic)')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def plot_confusion_matrix(y_test, y_pred, output_dir='.'):
    """
    Plota a matriz de confusão.
    
    Args:
        y_test: Valores reais.
        y_pred: Valores preditos.
        output_dir: Diretório para salvar a visualização.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def train_random_forest(X_train, y_train, param_grid=None):
    """
    Treina um modelo Random Forest com otimização de hiperparâmetros.
    
    Args:
        X_train: Dados de treino.
        y_train: Variáveis alvo de treino.
        param_grid: Grid de hiperparâmetros para otimização.
        
    Returns:
        model: Modelo treinado com os melhores hiperparâmetros.
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    # Inicializar o modelo base
    rf = RandomForestClassifier(random_state=42)
    
    # Configurar a validação cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Configurar a busca em grade com validação cruzada
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Ajustar a busca em grade aos dados
    grid_search.fit(X_train, y_train)
    
    # Imprimir os melhores hiperparâmetros
    print(f"Melhores hiperparâmetros: {grid_search.best_params_}")
    print(f"Melhor score de validação cruzada: {grid_search.best_score_:.4f}")
    
    # Retornar o melhor modelo
    return grid_search.best_estimator_

def train_svm(X_train, y_train, param_grid=None):
    """
    Treina um modelo SVM com otimização de hiperparâmetros.
    
    Args:
        X_train: Dados de treino.
        y_train: Variáveis alvo de treino.
        param_grid: Grid de hiperparâmetros para otimização.
        
    Returns:
        model: Modelo treinado com os melhores hiperparâmetros.
    """
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear', 'poly']
        }
    
    # Inicializar o modelo base
    svm = SVC(probability=True, random_state=42)
    
    # Configurar a validação cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Configurar a busca em grade com validação cruzada
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Ajustar a busca em grade aos dados
    grid_search.fit(X_train, y_train)
    
    # Imprimir os melhores hiperparâmetros
    print(f"Melhores hiperparâmetros: {grid_search.best_params_}")
    print(f"Melhor score de validação cruzada: {grid_search.best_score_:.4f}")
    
    # Retornar o melhor modelo
    return grid_search.best_estimator_

def train_naive_bayes(X_train, y_train):
    """
    Treina um modelo Naive Bayes.
    
    Args:
        X_train: Dados de treino.
        y_train: Variáveis alvo de treino.
        
    Returns:
        model: Modelo treinado.
    """
    # Inicializar e treinar o modelo
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, output_dir='.'):
    """
    Avalia o desempenho do modelo.
    
    Args:
        model: Modelo treinado.
        X_test: Dados de teste.
        y_test: Variáveis alvo de teste.
        output_dir: Diretório para salvar visualizações.
        
    Returns:
        dict: Métricas de desempenho.
    """
    # Fazer predições
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calcular AUC-ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Imprimir relatório de classificação
    print(classification_report(y_test, y_pred))
    
    # Plotar visualizações
    plot_confusion_matrix(y_test, y_pred, output_dir)
    plot_roc_curve(fpr, tpr, auc, output_dir)
    
    # Retornar métricas
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def save_model(model, scaler, feature_names, output_dir='models'):
    """
    Salva o modelo treinado e informações relacionadas.
    
    Args:
        model: Modelo treinado.
        scaler: Scaler usado para normalizar os dados.
        feature_names: Nomes das features.
        output_dir: Diretório para salvar o modelo.
    """
    # Criar diretório se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salvar o modelo
    model_file = os.path.join(output_dir, 'model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Salvar o scaler
    scaler_file = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Salvar os nomes das features
    feature_file = os.path.join(output_dir, 'feature_names.pkl')
    with open(feature_file, 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(f"Modelo, scaler e nomes de features salvos em {output_dir}")

def train_and_evaluate(data_file='data_processed.csv', output_dir='models'):
    """
    Treina e avalia modelos, salvando o melhor.
    
    Args:
        data_file: Caminho para o arquivo CSV com os dados processados.
        output_dir: Diretório para salvar o modelo e visualizações.
    """
    # Carregar dados
    X, y, feature_names = load_data(data_file)
    
    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalizar features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Definir modelos a serem treinados
    models = {
        'Random Forest': {
            'train_func': train_random_forest,
            'kwargs': {}
        },
        'SVM': {
            'train_func': train_svm,
            'kwargs': {}
        },
        'Naive Bayes': {
            'train_func': train_naive_bayes,
            'kwargs': {}
        }
    }
    
    # Armazenar resultados
    results = {}
    
    # Treinar e avaliar cada modelo
    for name, config in models.items():
        print(f"\n=== Treinando {name} ===")
        
        # Criar diretório específico para o modelo
        model_dir = os.path.join(output_dir, name.lower().replace(' ', '_'))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Treinar modelo
        model = config['train_func'](X_train_scaled, y_train, **config['kwargs'])
        
        # Avaliar modelo
        metrics = evaluate_model(model, X_test_scaled, y_test, model_dir)
        results[name] = metrics
        
        # Plotar importância das features (se aplicável)
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, feature_names, model_dir)
        
        # Salvar modelo
        save_model(model, scaler, feature_names, model_dir)
    
    # Identificar o melhor modelo com base no F1-score
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nMelhor modelo: {best_model[0]} (F1-score: {best_model[1]['f1']:.4f})")
    
    # Copiar o melhor modelo para o diretório principal
    best_model_dir = os.path.join(output_dir, best_model[0].lower().replace(' ', '_'))
    files_to_copy = ['model.pkl', 'scaler.pkl', 'feature_names.pkl']
    
    for file in files_to_copy:
        source = os.path.join(best_model_dir, file)
        destination = os.path.join(output_dir, file)
        
        with open(source, 'rb') as src_file, open(destination, 'wb') as dest_file:
            dest_file.write(src_file.read())
    
    print(f"Melhor modelo ({best_model[0]}) copiado para {output_dir}")
    
    return results

if __name__ == "__main__":
    # Verificar se o arquivo de dados processados existe
    if not os.path.exists('data_processed.csv'):
        print("Arquivo data_processed.csv não encontrado. Execute o módulo de processamento de dados primeiro.")
    else:
        # Treinar e avaliar modelos
        train_and_evaluate() 