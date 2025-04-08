#!/usr/bin/env python
"""
Main entry point for the Lead Scoring project.
Provides a command-line interface for the complete pipeline.
"""

import argparse
import logging
import logging.config
import sys
from pathlib import Path
import numpy as np

from src.config import LOGGING_CONFIG, RAW_DATA_FILE, INTERMEDIATE_DATA_FILE, FINAL_DATA_FILE
from src.data.pipeline import run_data_pipeline
from src.train import train_model
from src.predict import predict, Predictor

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def process_data(args):
    """Run the data processing pipeline."""
    logger.info("Starting data processing")
    try:
        # Process from raw to intermediate
        result1 = run_data_pipeline(
            input_file=RAW_DATA_FILE,
            output_file=INTERMEDIATE_DATA_FILE
        )
        logger.info(f"Processed raw data to intermediate: {result1['data'].shape}")

        # Process from intermediate to final
        result2 = run_data_pipeline(
            input_file=INTERMEDIATE_DATA_FILE,
            output_file=FINAL_DATA_FILE
        )
        logger.info(f"Processed intermediate data to final: {result2['data'].shape}")
        
        return True
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        return False

def train_models(args):
    """Train multiple machine learning models."""
    logger.info("Starting model training")
    results = {}
    
    model_types = args.models.split(',') if args.models else ['random_forest', 'xgboost', 'logistic_regression']
    
    try:
        for model_type in model_types:
            logger.info(f"Training {model_type} model")
            result = train_model(model_type)
            results[model_type] = result
            logger.info(f"{model_type} ROC AUC: {result['test_metrics']['roc_auc']:.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['test_metrics']['roc_auc'])
        logger.info(f"Best model: {best_model[0]} with ROC AUC: {best_model[1]['test_metrics']['roc_auc']:.4f}")
        
        return results
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return None

def make_predictions(args):
    """Make predictions using a trained model."""
    logger.info("Starting prediction")
    
    try:
        # Use specified model or default to random_forest
        model_type = args.model or 'random_forest'
        
        # Get predictions
        predictions = predict(
            data=Path(args.input),
            model_type=model_type,
            return_proba=True
        )
        
        # Save predictions if output file is specified
        if args.output:
            predictions.to_csv(args.output, index=False)
            logger.info(f"Saved predictions to {args.output}")
        
        # Show top leads
        top_n = min(args.top, len(predictions))
        top_leads = predictions.sort_values(by='prob_1', ascending=False).head(top_n)
        logger.info(f"Top {top_n} leads with highest conversion probability:")
        logger.info(top_leads)
        
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None

def full_pipeline(args):
    """Run the complete pipeline: process data, train models, and make predictions."""
    logger.info("Starting full pipeline")
    
    # Process data
    if not process_data(args):
        return False
    
    # Train models
    results = train_models(args)
    if not results:
        return False
    
    # Make predictions
    if args.input:
        make_predictions(args)
    
    logger.info("Full pipeline completed successfully")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Lead Scoring ML Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process data command
    process_parser = subparsers.add_parser("process", help="Process data")
    
    # Train models command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--models", type=str, help="Comma-separated list of models to train")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--model", type=str, help="Model type to use")
    predict_parser.add_argument("--input", type=str, required=True, help="Input data file")
    predict_parser.add_argument("--output", type=str, help="Output predictions file")
    predict_parser.add_argument("--top", type=int, default=5, help="Number of top leads to show")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--models", type=str, help="Comma-separated list of models to train")
    pipeline_parser.add_argument("--input", type=str, help="Input data file for predictions")
    pipeline_parser.add_argument("--output", type=str, help="Output predictions file")
    pipeline_parser.add_argument("--top", type=int, default=5, help="Number of top leads to show")
    
    args = parser.parse_args()
    
    if args.command == "process":
        return process_data(args)
    elif args.command == "train":
        return train_models(args) is not None
    elif args.command == "predict":
        return make_predictions(args) is not None
    elif args.command == "pipeline":
        return full_pipeline(args)
    else:
        parser.print_help()
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 