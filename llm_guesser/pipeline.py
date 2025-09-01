#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Age Prediction Pipeline

This module provides a complete pipeline for LLM-based age prediction,
integrating data loading, text formatting, and LLM prediction components.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import time
import json
from pathlib import Path

from dotenv import load_dotenv

from .data_loader import UserDataLoader
from .text_formatter import TextFormatter
from .llm_predictor import LLMPredictor, AgePrediction

# Load environment variables
load_dotenv()


class AgePredictionPipeline:
    """
    Complete pipeline for LLM-based age prediction.
    
    This class orchestrates the entire prediction process from raw data
    to final age predictions with evaluation metrics.
    """
    
    def __init__(self, 
                 user_data_path: str = "../data/user_data_public.csv",
                 question_data_path: str = "../data/question_data.csv",
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-1.5-flash"):
        """
        Initialize the prediction pipeline.
        
        Args:
            user_data_path: Path to user data CSV
            question_data_path: Path to question metadata CSV
            api_key: Google API key for Gemini
            model_name: Gemini model to use
        """
        # Initialize components
        self.data_loader = UserDataLoader(user_data_path, question_data_path)
        self.text_formatter = TextFormatter(self.data_loader)
        self.llm_predictor = LLMPredictor(api_key=api_key, model_name=model_name)
        
        print("Age Prediction Pipeline initialized successfully")
    
    def predict_ages(self, 
                    n_samples: Optional[int] = 10,
                    columns: Optional[List[str]] = None,
                    max_questions: int = 30,
                    min_age: int = 18,
                    max_age: int = 80,
                    include_demographics: bool = True,
                    delay_seconds: float = 0.5,
                    random_seed: int = 42) -> Tuple[List[AgePrediction], List[float], pd.DataFrame]:
        """
        Run the complete age prediction pipeline.
        
        Args:
            n_samples: Number of samples to predict (None for all)
            columns: List of specific columns to include (None for smart selection)
            max_questions: Maximum questions per user in prompt
            min_age: Minimum age filter
            max_age: Maximum age filter
            include_demographics: Include demographic info in prompts
            delay_seconds: Delay between API calls
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (predictions, true_ages, user_data)
        """
        print(f"\n{'='*60}")
        print("STARTING AGE PREDICTION PIPELINE")
        print(f"{'='*60}")
        
        # Step 1: Load and filter data
        print("\n1. Loading and filtering user data...")
        user_data, true_ages = self.data_loader.get_rows(
            columns=columns,
            n_samples=n_samples,
            min_age=min_age,
            max_age=max_age,
            random_seed=random_seed
        )
        
        print(f"Selected {len(user_data)} users for prediction")
        print(f"Age range: {true_ages.min():.0f} - {true_ages.max():.0f} years")
        
        # Step 2: Format user data into text
        print("\n2. Converting user data to text descriptions...")
        user_texts = self.text_formatter.format_batch(
            user_data,
            max_questions=max_questions,
            include_demographics=include_demographics,
            include_keywords=True,
            verbose=False
        )
        
        print(f"Generated {len(user_texts)} text descriptions")
        print(f"Average text length: {np.mean([len(text) for text in user_texts]):.0f} characters")
        
        # Step 3: Predict ages using LLM
        print("\n3. Predicting ages using LLM...")
        predictions = self.llm_predictor.predict_batch(
            user_texts,
            delay_seconds=delay_seconds
        )
        
        return predictions, list(true_ages), user_data
    
    # Removed progress callback since we're using tqdm now
    
    def evaluate_results(self, 
                        predictions: List[AgePrediction], 
                        true_ages: List[float]) -> Dict[str, Union[float, int, List]]:
        """
        Evaluate prediction results comprehensively.
        
        Args:
            predictions: List of age predictions
            true_ages: List of true ages
            
        Returns:
            Dictionary with detailed evaluation metrics
        """
        print(f"\n{'='*60}")
        print("EVALUATING PREDICTION RESULTS")
        print(f"{'='*60}")
        
        # Basic metrics
        basic_metrics = self.llm_predictor.evaluate_predictions(predictions, true_ages)
        
        # Additional analysis
        successful_preds = [p for p in predictions if p.success]
        failed_preds = [p for p in predictions if not p.success]
        
        # Confidence analysis
        if successful_preds:
            confidences = [p.confidence for p in successful_preds]
            conf_stats = {
                'mean_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences)
            }
        else:
            conf_stats = {
                'mean_confidence': 0.0,
                'std_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            }
        
        # Error analysis
        if successful_preds:
            valid_pairs = [
                (pred.predicted_age, true_age) 
                for pred, true_age in zip(predictions, true_ages) 
                if pred.success
            ]
            errors = [abs(p - a) for p, a in valid_pairs]
            
            error_stats = {
                'median_error': np.median(errors),
                'q25_error': np.percentile(errors, 25),
                'q75_error': np.percentile(errors, 75),
                'max_error': np.max(errors),
                'errors_over_10': sum(1 for e in errors if e > 10),
                'errors_over_20': sum(1 for e in errors if e > 20)
            }
        else:
            error_stats = {
                'median_error': float('inf'),
                'q25_error': float('inf'),
                'q75_error': float('inf'),
                'max_error': float('inf'),
                'errors_over_10': len(predictions),
                'errors_over_20': len(predictions)
            }
        
        # Compile comprehensive results
        results = {
            **basic_metrics,
            **conf_stats,
            **error_stats,
            'total_predictions': len(predictions),
            'successful_predictions': len(successful_preds),
            'failed_predictions': len(failed_preds),
            'failure_rate': len(failed_preds) / len(predictions) if predictions else 1.0
        }
        
        # Print results
        self._print_evaluation_results(results)
        
        return results
    
    def _print_evaluation_results(self, results: Dict):
        """Print evaluation results in a formatted way."""
        print("\n📊 PREDICTION ACCURACY:")
        print(f"   Mean Absolute Error (MAE): {results['mae']:.2f} years")
        print(f"   Root Mean Square Error (RMSE): {results['rmse']:.2f} years")
        print(f"   Mean Error (bias): {results['mean_error']:+.2f} years")
        print(f"   Median Error: {results['median_error']:.2f} years")
        
        print("\n🎯 ACCURACY THRESHOLDS:")
        print(f"   Within 5 years: {results['accuracy_5y']*100:.1f}%")
        print(f"   Within 10 years: {results['accuracy_10y']*100:.1f}%")
        print(f"   Errors > 10 years: {results['errors_over_10']}")
        print(f"   Errors > 20 years: {results['errors_over_20']}")
        
        print("\n🤖 MODEL PERFORMANCE:")
        print(f"   Success rate: {results['success_rate']*100:.1f}%")
        print(f"   Failed predictions: {results['failed_predictions']}")
        print(f"   Mean confidence: {results['mean_confidence']:.2f}")
        
        if results['mae'] < 10:
            print("\n✅ Good performance! MAE under 10 years.")
        elif results['mae'] < 15:
            print("\n⚠️  Moderate performance. Room for improvement.")
        else:
            print("\n❌ Poor performance. Consider model tuning.")
    
    def save_results(self, 
                    predictions: List[AgePrediction],
                    true_ages: List[float],
                    user_data: pd.DataFrame,
                    output_dir: str = "results") -> str:
        """
        Save prediction results to files.
        
        Args:
            predictions: Age predictions
            true_ages: True ages
            user_data: Original user data
            output_dir: Directory to save results
            
        Returns:
            Path to saved results directory
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Prepare results data
        results_data = []
        for i, (pred, true_age) in enumerate(zip(predictions, true_ages)):
            results_data.append({
                'user_id': i,
                'true_age': true_age,
                'predicted_age': pred.predicted_age,
                'confidence': pred.confidence,
                'error': abs(pred.predicted_age - true_age) if pred.success else None,
                'success': pred.success,
                'explanation': pred.explanation,
                'error_message': pred.error_message
            })
        
        # Save detailed results
        results_df = pd.DataFrame(results_data)
        results_file = output_path / f"predictions_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save evaluation metrics
        eval_results = self.evaluate_results(predictions, true_ages)
        eval_file = output_path / f"evaluation_{timestamp}.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Save sample user data (first 5 rows)
        sample_data = user_data.head(5).copy()
        sample_file = output_path / f"sample_users_{timestamp}.csv"
        sample_data.to_csv(sample_file, index=False)
        
        print(f"\n💾 Results saved to {output_path}/")
        print(f"   - {results_file.name}: Detailed predictions")
        print(f"   - {eval_file.name}: Evaluation metrics")  
        print(f"   - {sample_file.name}: Sample user data")
        
        return str(output_path)
    
    def run_complete_pipeline(self, 
                             n_samples: int = 20,
                             columns: Optional[List[str]] = None,
                             save_results: bool = True,
                             **kwargs) -> Dict:
        """
        Run the complete pipeline with evaluation and optional saving.
        
        Args:
            n_samples: Number of samples to process
            columns: List of specific columns to include (None for smart selection)
            save_results: Whether to save results to files
            **kwargs: Additional arguments for predict_ages()
            
        Returns:
            Dictionary with all results
        """
        start_time = time.time()
        
        # Run predictions
        predictions, true_ages, user_data = self.predict_ages(
            n_samples=n_samples,
            columns=columns,
            **kwargs
        )
        
        # Evaluate results
        evaluation = self.evaluate_results(predictions, true_ages)
        
        # Calculate total time
        total_time = time.time() - start_time
        evaluation['total_time_seconds'] = total_time
        evaluation['time_per_prediction'] = total_time / len(predictions)
        
        print(f"\n⏱️  Total time: {total_time:.1f} seconds")
        print(f"   Time per prediction: {total_time/len(predictions):.2f} seconds")
        
        # Save results if requested
        if save_results:
            results_dir = self.save_results(predictions, true_ages, user_data)
            evaluation['results_saved_to'] = results_dir
        
        return {
            'predictions': predictions,
            'true_ages': true_ages,
            'user_data': user_data,
            'evaluation': evaluation
        }


def main():
    """Example usage of the complete pipeline."""
    try:
        # Initialize pipeline
        pipeline = AgePredictionPipeline()
        
        # Run a small test
        print("Running pipeline test with 5 users...")
        results = pipeline.run_complete_pipeline(
            n_samples=5,
            max_questions=20,
            save_results=False  # Skip saving for test
        )
        
        print(f"\n🎉 Pipeline test completed successfully!")
        print(f"Processed {len(results['predictions'])} users")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        print("Make sure your Google API key is set and you have the required data files.")

if __name__ == "__main__":
    main()
