#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gemini-Based Age Prediction System

This system uses Google's Gemini LLM to predict user age (and other characteristics)
from OkCupid question responses, competing with traditional neural network models.

Features:
- Flexible question subset selection
- Batch processing with rate limiting
- Multiple prediction targets (age, gender, etc.)
- Performance comparison with existing NN models
- Advanced prompt engineering for demographic prediction
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from tqdm.auto import tqdm
import pickle
import re

# Statistical and ML imports
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Gemini API imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Import configuration
try:
    from gemini_config import *
except ImportError:
    print("Warning: gemini_config.py not found. Using default configuration.")
    GEMINI_API_KEY = "your_gemini_api_key_here"
    GEMINI_MODEL = "gemini-1.5-flash"
    DEFAULT_SAMPLE_SIZE = 1000
    MAX_QUESTIONS_PER_PROMPT = 50
    BATCH_SIZE = 10
    DEFAULT_TARGET = "d_age"
    RESULTS_DIR = "gemini_results"
    DATA_PATH = "data/user_data_public.csv"
    QUESTION_DATA_PATH = "data/question_data.csv"


class GeminiPredictor:
    """
    Main class for LLM-based demographic prediction using Gemini API.
    
    This class handles data preprocessing, prompt engineering, API calls,
    and evaluation for predicting user demographics from survey responses.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash",
                 temperature: float = 0.1, max_tokens: int = 100):
        """Initialize the Gemini predictor with API configuration."""
        
        # API Configuration
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Gemini
        self._setup_gemini()
        
        # Data storage
        self.user_data = None
        self.question_data = None
        self.selected_questions = None
        self.prediction_cache = {}
        
        # Results storage
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _setup_gemini(self):
        """Setup Gemini API client with safety settings."""
        if self.api_key == "your_gemini_api_key_here" or not self.api_key:
            print("Warning: No valid Gemini API key provided. Please set GEMINI_API_KEY in gemini_config.py")
            self.model = None
            return
            
        try:
            genai.configure(api_key=self.api_key)
            
            # Configure safety settings to be less restrictive for demographic questions
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_HIGH_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=safety_settings
            )
            
            print(f"Gemini API initialized successfully with model: {self.model_name}")
            
        except Exception as e:
            print(f"Error initializing Gemini API: {str(e)}")
            print("Please check your API key and internet connection.")
            self.model = None
    
    def load_data(self, data_path: str = None, question_data_path: str = None, 
                  sample_size: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess the OkCupid dataset.
        
        Args:
            data_path: Path to user data CSV
            question_data_path: Path to question metadata CSV  
            sample_size: Number of users to sample (None for all)
            
        Returns:
            Tuple of (user_data, question_data) DataFrames
        """
        data_path = data_path or DATA_PATH
        question_data_path = question_data_path or QUESTION_DATA_PATH
        sample_size = sample_size or DEFAULT_SAMPLE_SIZE
        
        print(f"Loading data from {data_path}")
        self.user_data = pd.read_csv(data_path, low_memory=False)
        
        print(f"Loading question metadata from {question_data_path}")
        self.question_data = pd.read_csv(question_data_path, delimiter=';', low_memory=False)
        
        # Basic data exploration
        print(f"User data shape: {self.user_data.shape}")
        print(f"Question data shape: {self.question_data.shape}")
        
        # Sample data if requested
        if sample_size and sample_size < len(self.user_data):
            print(f"Sampling {sample_size} users from {len(self.user_data)}")
            self.user_data = self.user_data.sample(sample_size, random_state=42)
        
        return self.user_data, self.question_data
    
    def select_questions(self, method: str = "high_response", n_questions: int = 30,
                        min_responses: int = 10000, custom_questions: List[str] = None) -> List[str]:
        """
        Select a subset of questions for prediction.
        
        Args:
            method: Selection method ("high_response", "age_predictive", "random", "custom")
            n_questions: Number of questions to select
            min_responses: Minimum number of responses for a question to be considered
            custom_questions: List of specific questions to use (for method="custom")
            
        Returns:
            List of selected question IDs
        """
        if custom_questions:
            method = "custom"
        
        if method == "custom" and custom_questions:
            # Use provided custom questions
            available_questions = [q for q in custom_questions if q in self.user_data.columns]
            print(f"Using {len(available_questions)} custom questions")
            self.selected_questions = available_questions
            
        elif method == "high_response":
            # Select questions with most responses
            if self.question_data is not None and 'N' in self.question_data.columns:
                # Use question metadata if available
                high_response_questions = self.question_data[
                    self.question_data['N'] >= min_responses
                ].sort_values('N', ascending=False).head(n_questions)
                
                # Get question IDs that exist in user data
                question_ids = high_response_questions.index.tolist()
                available_questions = [q for q in question_ids if q in self.user_data.columns]
            else:
                # Fallback: count non-null responses in user data
                question_cols = [col for col in self.user_data.columns if col.startswith('q')]
                response_counts = self.user_data[question_cols].notna().sum().sort_values(ascending=False)
                available_questions = response_counts[response_counts >= min_responses].head(n_questions).index.tolist()
            
            print(f"Selected {len(available_questions)} high-response questions")
            self.selected_questions = available_questions
            
        elif method == "age_predictive":
            # Select questions most predictive of age (requires some analysis)
            self.selected_questions = self._select_age_predictive_questions(n_questions)
            
        elif method == "random":
            # Random selection of questions
            question_cols = [col for col in self.user_data.columns if col.startswith('q')]
            available_questions = np.random.choice(question_cols, 
                                                 min(n_questions, len(question_cols)), 
                                                 replace=False).tolist()
            print(f"Selected {len(available_questions)} random questions")
            self.selected_questions = available_questions
            
        else:
            raise ValueError(f"Unknown question selection method: {method}")
        
        return self.selected_questions
    
    def _select_age_predictive_questions(self, n_questions: int) -> List[str]:
        """Select questions most predictive of age using statistical correlation."""
        print("Analyzing questions for age predictiveness...")
        
        # Get question columns and valid age data
        question_cols = [col for col in self.user_data.columns if col.startswith('q')]
        age_data = self.user_data['d_age'].dropna()
        
        correlations = {}
        
        for col in tqdm(question_cols, desc="Analyzing questions"):
            if col not in self.user_data.columns:
                continue
                
            # Get valid data for both question and age
            valid_data = self.user_data[[col, 'd_age']].dropna()
            
            if len(valid_data) < 50:  # Skip questions with too few responses
                continue
                
            try:
                # For categorical questions, use ANOVA F-statistic
                if self.user_data[col].dtype == 'object':
                    groups = [group['d_age'].values for name, group in valid_data.groupby(col)]
                    if len(groups) > 1:
                        f_stat, p_value = stats.f_oneway(*groups)
                        correlations[col] = f_stat if not np.isnan(f_stat) else 0
                else:
                    # For numeric questions, use correlation
                    corr = valid_data[col].corr(valid_data['d_age'])
                    correlations[col] = abs(corr) if not np.isnan(corr) else 0
                    
            except Exception:
                correlations[col] = 0
        
        # Select top questions
        sorted_questions = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected = [q[0] for q in sorted_questions[:n_questions]]
        
        print(f"Selected top {len(selected)} age-predictive questions")
        print(f"Top 5 questions: {selected[:5]}")
        
        return selected
    
    def create_prediction_prompt(self, user_responses: Dict[str, Any], target: str = "d_age") -> str:
        """
        Create a prompt for the LLM to predict user demographics.
        
        Args:
            user_responses: Dictionary of question responses for a user
            target: Target variable to predict
            
        Returns:
            Formatted prompt string
        """
        # Base prompt templates for different targets
        prompts = {
            "d_age": {
                "intro": "You are an expert demographic analyst. Based on the following survey responses from an online dating profile, predict the person's age.",
                "task": "Predict the person's age based on their responses. Respond with only a number between 18 and 100.",
                "format": "Age: [number]"
            },
            "d_gender": {
                "intro": "You are an expert demographic analyst. Based on the following survey responses from an online dating profile, predict the person's gender.",
                "task": "Predict the person's gender based on their responses. Respond with only: 'Man', 'Woman', or 'Non-binary'.",
                "format": "Gender: [Man/Woman/Non-binary]"
            },
            "d_orientation": {
                "intro": "You are an expert demographic analyst. Based on the following survey responses from an online dating profile, predict the person's sexual orientation.",
                "task": "Predict the person's sexual orientation. Respond with only: 'Straight', 'Gay', 'Bisexual', or 'Other'.",
                "format": "Orientation: [Straight/Gay/Bisexual/Other]"
            },
            "d_education_type": {
                "intro": "You are an expert demographic analyst. Based on the following survey responses from an online dating profile, predict the person's education level.",
                "task": "Predict the person's education level. Respond with: 'High School', 'Some College', 'College', 'Graduate School', or 'Other'.",
                "format": "Education: [level]"
            }
        }
        
        # Get prompt template or use age as default
        prompt_template = prompts.get(target, prompts["d_age"])
        
        # Build the prompt
        prompt_parts = [
            prompt_template["intro"],
            "",
            "Survey Responses:",
        ]
        
        # Add question responses with context when possible
        for question_id, response in user_responses.items():
            if pd.isna(response) or response == "" or str(response).lower() in ['nan', 'none']:
                continue
                
            # Try to get question text for context
            question_text = self._get_question_text(question_id)
            
            if question_text:
                prompt_parts.append(f"Q: {question_text}")
                prompt_parts.append(f"A: {response}")
            else:
                prompt_parts.append(f"{question_id}: {response}")
            
            prompt_parts.append("")
        
        # Add task and format instructions
        prompt_parts.extend([
            "",
            prompt_template["task"],
            f"Format your response exactly as: {prompt_template['format']}",
            "",
            "Your prediction:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_question_text(self, question_id: str) -> str:
        """Get the full text of a question for context in prompts."""
        if self.question_data is None:
            return None
            
        try:
            # Find question in metadata
            if question_id in self.question_data.index:
                question_row = self.question_data.loc[question_id]
                
                # Build question text
                text_parts = []
                if 'text' in question_row and pd.notna(question_row['text']):
                    text_parts.append(str(question_row['text']))
                
                # Add options if available
                options = []
                for i in range(1, 5):  # option_1 through option_4
                    option_col = f'option_{i}'
                    if option_col in question_row and pd.notna(question_row[option_col]):
                        options.append(str(question_row[option_col]))
                
                if options:
                    text_parts.append(f" Options: {', '.join(options)}")
                
                return "".join(text_parts)
                
        except Exception:
            pass
            
        return None
    
    def predict_single(self, user_responses: Dict[str, Any], target: str = "d_age") -> Optional[Union[int, str]]:
        """
        Make a single prediction using the Gemini API.
        
        Args:
            user_responses: Dictionary of user's question responses
            target: Target variable to predict
            
        Returns:
            Predicted value or None if prediction failed
        """
        if not self.model:
            print("Gemini model not initialized. Cannot make predictions.")
            return None
        
        # Create cache key
        cache_key = hash(str(sorted(user_responses.items())) + target)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        try:
            # Create prompt
            prompt = self.create_prediction_prompt(user_responses, target)
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            # Extract prediction from response
            prediction = self._extract_prediction(response.text, target)
            
            # Cache result
            self.prediction_cache[cache_key] = prediction
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None
    
    def _extract_prediction(self, response_text: str, target: str) -> Optional[Union[int, str]]:
        """Extract the actual prediction value from the LLM response."""
        if not response_text:
            return None
        
        response_text = response_text.strip()
        
        try:
            if target == "d_age":
                # Extract age number
                age_match = re.search(r'(?:Age:?\s*)?(\d+)', response_text)
                if age_match:
                    age = int(age_match.group(1))
                    # Clamp to reasonable range
                    return max(18, min(100, age))
                
            elif target == "d_gender":
                # Extract gender
                text_lower = response_text.lower()
                if 'woman' in text_lower or 'female' in text_lower:
                    return 'Woman'
                elif 'man' in text_lower or 'male' in text_lower:
                    return 'Man'
                elif 'non-binary' in text_lower or 'nonbinary' in text_lower:
                    return 'Non-binary'
                    
            elif target == "d_orientation":
                # Extract orientation
                text_lower = response_text.lower()
                if 'straight' in text_lower or 'heterosexual' in text_lower:
                    return 'Straight'
                elif 'gay' in text_lower or 'homosexual' in text_lower:
                    return 'Gay'
                elif 'bisexual' in text_lower or 'bi' in text_lower:
                    return 'Bisexual'
                else:
                    return 'Other'
                    
            # For other targets, try to extract first meaningful word/number
            else:
                # Try to find any meaningful response
                words = response_text.split()
                if words:
                    return words[0]
                    
        except Exception as e:
            print(f"Error extracting prediction from '{response_text}': {e}")
            
        return None
    
    def predict_batch(self, users_data: pd.DataFrame, target: str = "d_age", 
                     batch_size: int = None, delay: float = 1.0) -> List[Optional[Union[int, str]]]:
        """
        Make predictions for a batch of users with rate limiting.
        
        Args:
            users_data: DataFrame containing user data
            target: Target variable to predict
            batch_size: Number of requests per batch
            delay: Delay between API calls (seconds)
            
        Returns:
            List of predictions
        """
        batch_size = batch_size or BATCH_SIZE
        predictions = []
        
        print(f"Making predictions for {len(users_data)} users (target: {target})")
        print(f"Batch size: {batch_size}, Delay: {delay}s")
        
        for i in tqdm(range(0, len(users_data), batch_size)):
            batch_data = users_data.iloc[i:i+batch_size]
            batch_predictions = []
            
            for _, user in batch_data.iterrows():
                # Extract user responses for selected questions
                user_responses = {}
                for question in self.selected_questions:
                    if question in user.index and pd.notna(user[question]):
                        user_responses[question] = user[question]
                
                # Make prediction
                prediction = self.predict_single(user_responses, target)
                batch_predictions.append(prediction)
                
                # Rate limiting delay
                if delay > 0:
                    time.sleep(delay)
            
            predictions.extend(batch_predictions)
            
            # Progress update
            if i % (batch_size * 5) == 0:
                print(f"Processed {i + len(batch_data)} users...")
        
        return predictions
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: List, target: str = "d_age") -> Dict[str, float]:
        """
        Evaluate prediction performance with appropriate metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            target: Target variable name
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Filter out None predictions
        valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
        
        if not valid_indices:
            print("No valid predictions to evaluate!")
            return {}
        
        y_true_valid = y_true[valid_indices]
        y_pred_valid = np.array([y_pred[i] for i in valid_indices])
        
        print(f"\nEvaluation Results ({target}):")
        print(f"Valid predictions: {len(valid_indices)}/{len(y_pred)} ({len(valid_indices)/len(y_pred)*100:.1f}%)")
        
        metrics = {}
        
        if target == "d_age":
            # Regression metrics for age
            mae = mean_absolute_error(y_true_valid, y_pred_valid)
            
            # Calculate accuracy within different year thresholds
            abs_errors = np.abs(y_true_valid - y_pred_valid)
            within_1_year = np.mean(abs_errors <= 1) * 100
            within_3_years = np.mean(abs_errors <= 3) * 100
            within_5_years = np.mean(abs_errors <= 5) * 100
            
            metrics = {
                'mae': mae,
                'within_1_year': within_1_year,
                'within_3_years': within_3_years,
                'within_5_years': within_5_years,
                'valid_predictions': len(valid_indices) / len(y_pred)
            }
            
            print(f"Mean Absolute Error: {mae:.2f} years")
            print(f"Within ±1 year: {within_1_year:.1f}%")
            print(f"Within ±3 years: {within_3_years:.1f}%")
            print(f"Within ±5 years: {within_5_years:.1f}%")
            
        else:
            # Classification metrics for categorical targets
            accuracy = accuracy_score(y_true_valid, y_pred_valid)
            
            metrics = {
                'accuracy': accuracy,
                'valid_predictions': len(valid_indices) / len(y_pred)
            }
            
            print(f"Accuracy: {accuracy:.3f}")
            
            # Print classification report
            try:
                print("\nClassification Report:")
                print(classification_report(y_true_valid, y_pred_valid))
            except Exception as e:
                print(f"Could not generate classification report: {e}")
        
        return metrics
    
    def compare_with_baselines(self, y_true: np.ndarray, y_pred: List, target: str = "d_age") -> Dict[str, float]:
        """
        Compare LLM predictions with simple baseline models.
        
        Args:
            y_true: True target values
            y_pred: LLM predictions
            target: Target variable
            
        Returns:
            Dictionary with comparison results
        """
        # Filter valid predictions
        valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
        y_true_valid = y_true[valid_indices]
        y_pred_valid = np.array([y_pred[i] for i in valid_indices])
        
        comparison_results = {}
        
        if target == "d_age":
            # Age baselines
            
            # Mean baseline
            mean_age = np.mean(y_true_valid)
            mean_predictions = np.full_like(y_true_valid, mean_age)
            mean_mae = mean_absolute_error(y_true_valid, mean_predictions)
            
            # Median baseline  
            median_age = np.median(y_true_valid)
            median_predictions = np.full_like(y_true_valid, median_age)
            median_mae = mean_absolute_error(y_true_valid, median_predictions)
            
            # LLM performance
            llm_mae = mean_absolute_error(y_true_valid, y_pred_valid)
            
            comparison_results = {
                'llm_mae': llm_mae,
                'mean_baseline_mae': mean_mae,
                'median_baseline_mae': median_mae,
                'improvement_over_mean': (mean_mae - llm_mae) / mean_mae * 100,
                'improvement_over_median': (median_mae - llm_mae) / median_mae * 100
            }
            
            print(f"\nBaseline Comparison:")
            print(f"LLM MAE: {llm_mae:.2f} years")
            print(f"Mean baseline MAE: {mean_mae:.2f} years")
            print(f"Median baseline MAE: {median_mae:.2f} years")
            print(f"Improvement over mean: {comparison_results['improvement_over_mean']:.1f}%")
            print(f"Improvement over median: {comparison_results['improvement_over_median']:.1f}%")
            
        else:
            # Classification baselines
            
            # Most frequent class baseline
            most_frequent = stats.mode(y_true_valid)[0][0]
            most_frequent_predictions = np.full_like(y_true_valid, most_frequent)
            most_frequent_accuracy = accuracy_score(y_true_valid, most_frequent_predictions)
            
            # LLM performance
            llm_accuracy = accuracy_score(y_true_valid, y_pred_valid)
            
            comparison_results = {
                'llm_accuracy': llm_accuracy,
                'most_frequent_baseline': most_frequent_accuracy,
                'improvement_over_baseline': (llm_accuracy - most_frequent_accuracy) / most_frequent_accuracy * 100
            }
            
            print(f"\nBaseline Comparison:")
            print(f"LLM Accuracy: {llm_accuracy:.3f}")
            print(f"Most frequent baseline: {most_frequent_accuracy:.3f}")
            print(f"Improvement: {comparison_results['improvement_over_baseline']:.1f}%")
        
        return comparison_results
    
    def save_results(self, results: Dict[str, Any], experiment_name: str = None) -> str:
        """
        Save experiment results to files.
        
        Args:
            results: Dictionary containing all results
            experiment_name: Name for the experiment
            
        Returns:
            Path to saved results directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_name or f"gemini_experiment_{timestamp}"
        
        exp_dir = os.path.join(self.results_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save main results as JSON
        results_file = os.path.join(exp_dir, "results.json")
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[key] = value.item()
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save configuration
        config_file = os.path.join(exp_dir, "config.json")
        config = {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'selected_questions': self.selected_questions,
            'timestamp': timestamp
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save predictions cache
        cache_file = os.path.join(exp_dir, "prediction_cache.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.prediction_cache, f)
        
        print(f"\nResults saved to: {exp_dir}")
        return exp_dir
    
    def create_visualizations(self, y_true: np.ndarray, y_pred: List, target: str = "d_age", 
                            save_dir: str = None) -> None:
        """
        Create visualizations for prediction results.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            target: Target variable
            save_dir: Directory to save plots
        """
        # Filter valid predictions
        valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
        y_true_valid = y_true[valid_indices]
        y_pred_valid = np.array([y_pred[i] for i in valid_indices])
        
        save_dir = save_dir or self.results_dir
        
        if target == "d_age":
            # Age prediction visualizations
            
            # Scatter plot: True vs Predicted
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.scatter(y_true_valid, y_pred_valid, alpha=0.6)
            
            # Perfect prediction line
            min_age = min(min(y_true_valid), min(y_pred_valid))
            max_age = max(max(y_true_valid), max(y_pred_valid))
            plt.plot([min_age, max_age], [min_age, max_age], 'r--', label='Perfect Prediction')
            
            plt.xlabel('True Age')
            plt.ylabel('Predicted Age')
            plt.title('Gemini Age Predictions vs True Age')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Error distribution
            plt.subplot(1, 2, 2)
            errors = y_pred_valid - y_true_valid
            plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
            plt.axvline(np.mean(errors), color='green', linestyle='--', label=f'Mean Error: {np.mean(errors):.1f}')
            plt.xlabel('Prediction Error (years)')
            plt.ylabel('Frequency')
            plt.title('Age Prediction Error Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'age_predictions_analysis.png'), dpi=300, bbox_inches='tight')
            
            plt.show()
        
        else:
            # Classification visualization
            from sklearn.metrics import confusion_matrix
            
            try:
                cm = confusion_matrix(y_true_valid, y_pred_valid)
                labels = sorted(set(y_true_valid) | set(y_pred_valid))
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=labels, yticklabels=labels)
                plt.title(f'Confusion Matrix: {target}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'{target}_confusion_matrix.png'), 
                               dpi=300, bbox_inches='tight')
                
                plt.show()
                
            except Exception as e:
                print(f"Could not create confusion matrix: {e}")


def run_experiment(api_key: str = None, target: str = "d_age", n_questions: int = 30,
                  question_selection: str = "high_response", sample_size: int = None,
                  custom_questions: List[str] = None, model_name: str = "gemini-1.5-flash") -> Dict[str, Any]:
    """
    Run a complete prediction experiment.
    
    Args:
        api_key: Gemini API key
        target: Target variable to predict
        n_questions: Number of questions to use
        question_selection: Method for selecting questions
        sample_size: Number of users to sample
        custom_questions: Custom list of questions
        model_name: Gemini model to use
        
    Returns:
        Dictionary containing all experiment results
    """
    print(f"=== Gemini Age Prediction Experiment ===")
    print(f"Target: {target}")
    print(f"Questions: {n_questions} ({question_selection})")
    print(f"Model: {model_name}")
    print(f"Sample size: {sample_size}")
    
    # Initialize predictor
    predictor = GeminiPredictor(api_key=api_key, model_name=model_name)
    
    # Load data
    predictor.load_data(sample_size=sample_size)
    
    # Select questions
    predictor.select_questions(
        method=question_selection,
        n_questions=n_questions,
        custom_questions=custom_questions
    )
    
    print(f"Selected {len(predictor.selected_questions)} questions")
    
    # Filter users with valid target data
    valid_users = predictor.user_data.dropna(subset=[target])
    print(f"Users with valid {target}: {len(valid_users)}")
    
    # Split data (optional - you might want to use specific test sets)
    train_data, test_data = train_test_split(valid_users, test_size=0.2, random_state=42)
    
    print(f"Using {len(test_data)} users for testing")
    
    # Make predictions
    predictions = predictor.predict_batch(test_data, target=target)
    true_values = test_data[target].values
    
    # Evaluate results
    metrics = predictor.evaluate_predictions(true_values, predictions, target)
    baseline_comparison = predictor.compare_with_baselines(true_values, predictions, target)
    
    # Combine results
    results = {
        'target': target,
        'n_questions': len(predictor.selected_questions),
        'question_selection_method': question_selection,
        'model_name': model_name,
        'sample_size': len(test_data),
        'selected_questions': predictor.selected_questions,
        'predictions': predictions,
        'true_values': true_values,
        'metrics': metrics,
        'baseline_comparison': baseline_comparison,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    exp_name = f"{target}_{question_selection}_{n_questions}q_{model_name.replace('-', '_')}"
    save_dir = predictor.save_results(results, exp_name)
    
    # Create visualizations
    predictor.create_visualizations(true_values, predictions, target, save_dir)
    
    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Gemini-based Age Prediction System')
    
    parser.add_argument('--api_key', type=str, help='Gemini API key')
    parser.add_argument('--target', type=str, default='d_age', 
                       help='Target variable to predict (d_age, d_gender, etc.)')
    parser.add_argument('--n_questions', type=int, default=30,
                       help='Number of questions to use')
    parser.add_argument('--question_selection', type=str, default='high_response',
                       choices=['high_response', 'age_predictive', 'random', 'custom'],
                       help='Method for selecting questions')
    parser.add_argument('--custom_questions', type=str, nargs='*',
                       help='Custom list of questions to use')
    parser.add_argument('--sample_size', type=int, default=200,
                       help='Number of users to sample for testing')
    parser.add_argument('--model', type=str, default='gemini-1.5-flash',
                       help='Gemini model to use')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_experiment(
        api_key=args.api_key,
        target=args.target,
        n_questions=args.n_questions,
        question_selection=args.question_selection,
        custom_questions=args.custom_questions,
        sample_size=args.sample_size,
        model_name=args.model
    )
    
    print("\n=== Experiment Complete ===")
    if 'mae' in results['metrics']:
        print(f"Final MAE: {results['metrics']['mae']:.2f} years")
    elif 'accuracy' in results['metrics']:
        print(f"Final Accuracy: {results['metrics']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
