#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example Usage of Gemini Age Prediction System

This script demonstrates how to use the Gemini-based age prediction system
with different configurations and compare it to neural network models.

Run this after setting up your API key in gemini_config.py
"""

import os
import sys
import pandas as pd
from gemini_age_predictor import GeminiPredictor, run_experiment
from model_comparison import run_full_comparison


def basic_age_prediction_example():
    """
    Basic example: Predict age using 20 high-response questions with Gemini.
    """
    print("=== Basic Age Prediction Example ===")
    
    # Initialize the predictor
    predictor = GeminiPredictor()
    
    # Load a small sample of data for testing
    try:
        predictor.load_data(sample_size=50)  # Small sample for quick testing
    except FileNotFoundError:
        print("Error: Could not find data files. Please ensure:")
        print("- data/user_data_public.csv exists")
        print("- data/question_data.csv exists")
        return None
    
    # Select 20 questions with highest response rates
    predictor.select_questions(method="high_response", n_questions=20)
    
    print(f"Selected questions: {predictor.selected_questions[:5]}...")  # Show first 5
    
    # Get a small test sample
    valid_users = predictor.user_data.dropna(subset=['d_age']).head(10)
    print(f"Testing on {len(valid_users)} users")
    
    # Make predictions
    predictions = predictor.predict_batch(
        valid_users, 
        target='d_age',
        batch_size=3,
        delay=0.5  # Shorter delay for demo
    )
    
    # Show results
    true_ages = valid_users['d_age'].values
    
    print("\n=== Predictions vs True Ages ===")
    for i, (true_age, predicted_age) in enumerate(zip(true_ages, predictions)):
        if predicted_age is not None:
            error = abs(true_age - predicted_age)
            print(f"User {i+1}: True={true_age:.0f}, Predicted={predicted_age:.0f}, Error={error:.1f}")
        else:
            print(f"User {i+1}: True={true_age:.0f}, Predicted=None (failed)")
    
    # Calculate basic metrics
    valid_predictions = [(true, pred) for true, pred in zip(true_ages, predictions) if pred is not None]
    
    if valid_predictions:
        valid_true, valid_pred = zip(*valid_predictions)
        mae = sum(abs(t - p) for t, p in valid_predictions) / len(valid_predictions)
        print(f"\nMean Absolute Error: {mae:.2f} years")
        print(f"Success rate: {len(valid_predictions)}/{len(true_ages)} ({len(valid_predictions)/len(true_ages)*100:.1f}%)")
    
    return predictor


def custom_questions_example():
    """
    Example using specific custom questions that might be particularly predictive.
    """
    print("\n=== Custom Questions Example ===")
    
    # Define custom questions that might be age-related
    custom_questions = [
        'q35',   # Sex vs Love preference
        'q41',   # Importance of religion 
        'q48',   # Normal vs Weird preference
        'q46',   # Good vs Interesting things
        'q77',   # Drug use frequency
        'q80',   # Drug use experience
    ]
    
    # Run experiment with custom questions
    results = run_experiment(
        target='d_age',
        n_questions=len(custom_questions),
        question_selection='custom',
        custom_questions=custom_questions,
        sample_size=30,  # Small sample
        model_name='gemini-1.5-flash'
    )
    
    if 'metrics' in results and 'mae' in results['metrics']:
        print(f"Custom Questions MAE: {results['metrics']['mae']:.2f} years")
        print(f"Questions used: {custom_questions}")
    
    return results


def predict_different_targets_example():
    """
    Example showing how to predict different targets besides age.
    """
    print("\n=== Predicting Different Targets ===")
    
    predictor = GeminiPredictor()
    
    try:
        predictor.load_data(sample_size=20)
    except FileNotFoundError:
        print("Data files not found - skipping this example")
        return
    
    # Select questions
    predictor.select_questions(method="high_response", n_questions=15)
    
    # Get test sample
    test_sample = predictor.user_data.head(5)
    
    # Try predicting different targets
    targets_to_try = ['d_age', 'd_gender', 'd_orientation']
    
    for target in targets_to_try:
        if target not in test_sample.columns:
            print(f"Target {target} not available in data")
            continue
            
        print(f"\n--- Predicting {target} ---")
        
        # Filter users with valid target data
        valid_users = test_sample.dropna(subset=[target])
        
        if len(valid_users) == 0:
            print(f"No users with valid {target} data")
            continue
        
        # Make predictions
        predictions = predictor.predict_batch(
            valid_users,
            target=target,
            batch_size=2,
            delay=0.3
        )
        
        # Show results
        true_values = valid_users[target].values
        
        for i, (true_val, pred_val) in enumerate(zip(true_values, predictions)):
            print(f"User {i+1}: True={true_val}, Predicted={pred_val}")


def model_comparison_example():
    """
    Example of comparing LLM performance with neural network models.
    """
    print("\n=== Model Comparison Example ===")
    print("This will compare Gemini LLM with simulated NN performance")
    print("(Based on the performance metrics from your existing models)")
    
    try:
        # Run comparison with a small sample
        results = run_full_comparison(sample_size=50)  # Small for demonstration
        
        if results and 'model_metrics' in results:
            print("\n=== Comparison Results ===")
            
            # Sort models by MAE
            models_by_performance = sorted(
                results['model_metrics'].items(),
                key=lambda x: x[1]['mae']
            )
            
            print("Models ranked by performance (MAE):")
            for i, (model_name, metrics) in enumerate(models_by_performance, 1):
                model_type = "Neural Network" if metrics['model_type'] == 'neural_network' else "LLM"
                print(f"{i}. {model_name} ({model_type}): {metrics['mae']:.2f} years MAE")
            
            # Find best of each type
            nn_models = {name: metrics for name, metrics in results['model_metrics'].items() 
                        if metrics['model_type'] == 'neural_network'}
            llm_models = {name: metrics for name, metrics in results['model_metrics'].items() 
                         if metrics['model_type'] == 'llm'}
            
            if nn_models and llm_models:
                best_nn = min(nn_models.items(), key=lambda x: x[1]['mae'])
                best_llm = min(llm_models.items(), key=lambda x: x[1]['mae'])
                
                print(f"\nBest Neural Network: {best_nn[0]} ({best_nn[1]['mae']:.2f} years)")
                print(f"Best LLM: {best_llm[0]} ({best_llm[1]['mae']:.2f} years)")
                
                difference = best_nn[1]['mae'] - best_llm[1]['mae']
                if difference > 0:
                    print(f"LLM is {difference:.2f} years better than NN")
                else:
                    print(f"NN is {abs(difference):.2f} years better than LLM")
        
    except Exception as e:
        print(f"Comparison failed: {str(e)}")
        print("This might be due to API issues or missing configuration")


def main():
    """
    Main function that runs all examples.
    
    Comment out any examples you don't want to run.
    """
    print("Gemini Age Prediction System - Example Usage")
    print("=" * 50)
    
    # Check if API key is set
    from gemini_config import GEMINI_API_KEY
    
    if GEMINI_API_KEY == "your_gemini_api_key_here":
        print("⚠️  WARNING: Please set your Gemini API key in gemini_config.py")
        print("⚠️  The examples will not work without a valid API key")
        print()
        response = input("Continue anyway to see the code structure? (y/n): ")
        if response.lower() != 'y':
            return
    
    try:
        # Run examples (comment out any you don't want to run)
        
        # 1. Basic age prediction
        predictor = basic_age_prediction_example()
        
        # 2. Custom questions
        if predictor:  # Only run if basic example worked
            custom_results = custom_questions_example()
        
        # 3. Different targets
        predict_different_targets_example()
        
        # 4. Model comparison (this will take longer)
        # model_comparison_example()  # Uncomment to run
        
        print("\n" + "=" * 50)
        print("Examples completed!")
        print("\nNext steps:")
        print("1. Set your Gemini API key in gemini_config.py")
        print("2. Adjust sample sizes and question counts for your needs")
        print("3. Use model_comparison.py for comprehensive evaluation")
        print("4. Modify prompts in gemini_age_predictor.py for better performance")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\nExample failed with error: {str(e)}")
        print("Check your data files and API configuration")


if __name__ == "__main__":
    main()
