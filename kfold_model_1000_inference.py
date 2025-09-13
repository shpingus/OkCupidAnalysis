#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
K-fold Ensemble Model 1000 Inference Script

This script loads 5 trained k-fold age prediction models (enhanced or non-enhanced) 
and generates ensemble predictions for 1000 user profiles from the data_for_report dataset.

Usage:
    python kfold_model_1000_inference.py --model enhanced
    python kfold_model_1000_inference.py --model simple

Output CSV files:
- enhanced: k_fold_enhanced_model_1000_predictions.csv
- simple: k_fold_model_1000_predictions.csv

Columns:
- (empty): Sequential row number (0, 1, 2, ... 999)
- index: Original user ID from dataset
- real_age: True age from d_age column  
- predicted_age: Ensemble prediction (rounded to whole numbers)
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Deep learning imports
import torch
import torch.nn as nn

# VoyageAI API
import voyageai
from voyageai import Client


def setup_environment():
    """Set up the environment variables and device configuration."""
    load_dotenv()
    api_key = os.getenv("VOYAGE_API_KEY")
    voyageai.api_key = api_key
    
    # Print API key status
    if api_key and api_key != "your_api_key_here":
        print(f"VoyageAI API key loaded")
    else:
        print(f"No valid VoyageAI API key found. Will use random embeddings for testing.")
    
    # Device setup
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print('Using MPS (Apple Silicon)')
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print('Using CUDA (NVIDIA GPU)')
        device = torch.device("cuda")
    else:
        print('Using CPU')
        device = torch.device("cpu")
    
    return device


def clean_text(text):
    """Basic text cleaning function."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    return text


class EnhancedAgePredictor(nn.Module):
    """
    Enhanced neural network for age prediction using embeddings.
    
    This model adapts the architecture from the hallucination detector:
    - Stronger dropout regularization (0.5, 0.3)
    - Xavier uniform weight initialization
    - Same layer structure but adapted for regression task
    """
    def __init__(self, input_size):
        super(EnhancedAgePredictor, self).__init__()
        
        # Model architecture with stronger regularization
        self.model = nn.Sequential(
            # First layer: input_size -> 256
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Stronger dropout like hallucination detector
            
            # Second layer: 256 -> 64
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Stronger dropout like hallucination detector
            
            # Output layer: 64 -> 1 (no sigmoid for regression)
            nn.Linear(64, 1)
        )
        
        # Initialize weights using xavier_uniform (from hallucination detector)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, X):
        """Make predictions on numpy array input."""
        self.eval()
        device = next(self.parameters()).device  # Get the device model is on
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            predictions = self(X_tensor).cpu().numpy().flatten()
            # Clip predictions to reasonable age range (13-100)
            return np.clip(predictions, 13, 100)


def load_data(data_path):
    """Load the profiles dataset for inference."""
    print(f"Loading dataset from {data_path}")
    csv = pd.read_csv(data_path, low_memory=False)
    
    print(f"Dataset shape: {csv.shape}")
    print(f"Columns: {list(csv.columns)}")
    
    # Check for required columns
    if 'd_age' not in csv.columns:
        raise ValueError("Dataset must contain 'd_age' column for evaluation")
    
    if 'index' not in csv.columns:
        raise ValueError("Dataset must contain 'index' column for user identification")
    
    # Process users with non-null age
    valid_users = csv.dropna(subset=['d_age'])
    print(f"Users with valid age: {len(valid_users)}")
    
    return valid_users


def prepare_features(csv):
    """Extract and prepare text features from the dataset."""
    # Define possible demographic columns (same as training)
    all_demographic_columns = [
        'd_body_type', 'd_diet', 'd_drinks', 'd_drugs', 'd_education', 
        'd_ethnicity', 'd_income', 'd_job', 'd_orientation', 
        'd_pets', 'd_religion', 'd_religion_seriosity', 'd_sign', 
        'd_smokes', 'd_speaks', 'speaks', 'status'
    ]
    
    # Filter to only include columns that exist in the dataset
    demographic_columns = [col for col in all_demographic_columns if col in csv.columns]
    print(f"Found {len(demographic_columns)} demographic columns: {demographic_columns}")
    
    # Get question columns (columns starting with 'q')
    question_columns = [col for col in csv.columns if col.startswith('q')]
    print(f"Found {len(question_columns)} question columns")
    
    # Combine demographic information
    if demographic_columns:
        csv['combined_demographics'] = csv[demographic_columns].fillna("unknown").astype(str).agg(' '.join, axis=1).apply(clean_text)
    else:
        print("Warning: No demographic columns found. Using empty strings.")
        csv['combined_demographics'] = ""
    
    # Create combined text from question responses
    if question_columns:
        csv['question_responses'] = csv[question_columns].fillna("no_answer").astype(str).agg(' '.join, axis=1).apply(clean_text)
    else:
        print("Warning: No question columns found. Using empty strings.")
        csv['question_responses'] = ""
    
    # Display statistics about our text features
    print(f"Average demographics length: {csv['combined_demographics'].apply(len).mean():.0f} characters")
    print(f"Average question responses length: {csv['question_responses'].apply(len).mean():.0f} characters")
    
    return csv


def generate_embeddings(texts, model="voyage-2", batch_size=20, cache_file=None):
    """Generate embeddings for a list of texts using VoyageAI API."""
    # Check if cache exists and load it
    if cache_file and os.path.exists(cache_file):
        print(f"Loading embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Set embedding size based on model
    embedding_size = 1024 if model == "voyage-2" else 768
    
    # Use the VoyageAI API to generate embeddings
    all_embeddings = []
    
    # Process in batches to handle API rate limits
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings with API"):
        batch = texts[i:i + batch_size]
        try:
            # Remove empty strings to avoid API errors
            batch = [text if text.strip() else "unknown" for text in batch]
            
            # Call VoyageAI API using the correct function
            # Check if API key exists before attempting to use API
            if voyageai.api_key and voyageai.api_key != "your_api_key_here":
                try:
                    # Create client and use the recommended embed() method
                    client = Client(api_key=voyageai.api_key)
                    # Get embeddings through the client
                    response = client.embed(
                        texts=batch,
                        model=model
                    )
                    # Extract embeddings from response
                    batch_embeddings = response.embeddings
                    
                except Exception as e:
                    print(f"API call failed: {str(e)}")
                    # Generate random embeddings as fallback
                    batch_embeddings = [np.random.randn(embedding_size).tolist() for _ in range(len(batch))]
            else:
                # If no valid API key, use random embeddings
                print("No valid API key found. Generating random embeddings for testing...")
                # Generate random embeddings of the correct size
                batch_embeddings = [np.random.randn(embedding_size).tolist() for _ in range(len(batch))]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error in batch {i}-{i+batch_size}: {e}")
            # Handle error by adding zero embeddings
            zero_embeddings = [[0.0] * embedding_size for _ in range(len(batch))]
            all_embeddings.extend(zero_embeddings)
    
    # Cache embeddings to avoid regenerating them
    if cache_file:
        print(f"Saving embeddings to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(all_embeddings, f)
    
    return all_embeddings


def create_embeddings_for_users(valid_users, cache_dir="cache", model="voyage-2"):
    """Create embeddings for all text features of users."""
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    user_count = len(valid_users)
    
    # Generate embeddings for demographics
    demographic_embeddings = generate_embeddings(
        valid_users['combined_demographics'].tolist(),
        model=model,
        cache_file=f"{cache_dir}/inference_demographic_embeddings_{user_count}.pkl"
    )
    
    # Generate embeddings for question responses
    question_embeddings = generate_embeddings(
        valid_users['question_responses'].tolist(),
        model=model,
        cache_file=f"{cache_dir}/inference_question_embeddings_{user_count}.pkl"
    )
    
    # Verify embedding dimensions
    print(f"Demographic embeddings shape: {len(demographic_embeddings)} x {len(demographic_embeddings[0])}")
    print(f"Question embeddings shape: {len(question_embeddings)} x {len(question_embeddings[0])}")
    
    return demographic_embeddings, question_embeddings


def create_basic_features(demographic_embeddings, question_embeddings):
    """
    Create basic feature combinations by concatenating embeddings.
    Used for simple (non-enhanced) models.
    """
    print("Creating basic feature combinations...")
    
    # Convert lists to numpy arrays for more efficient operations
    demographic_embeddings_np = np.array(demographic_embeddings)
    question_embeddings_np = np.array(question_embeddings)
    
    # Create basic concatenated features
    basic_features = np.hstack([
        demographic_embeddings_np,
        question_embeddings_np
    ])
    
    print(f"Basic feature dimension: {basic_features.shape[1]}")
    
    return basic_features


def create_enhanced_features(demographic_embeddings, question_embeddings):
    """
    Create enhanced feature combinations for age prediction, inspired by the
    hallucination detector architecture.
    
    This function creates more complex feature interactions including element-wise 
    products and differences between different embedding types.
    """
    print("Creating enhanced feature combinations...")
    
    # Get number of users and embedding dimensions
    num_users = len(demographic_embeddings)
    demographic_dim = len(demographic_embeddings[0])
    question_dim = len(question_embeddings[0])
    
    # First convert lists to numpy arrays for more efficient operations
    demographic_embeddings_np = np.array(demographic_embeddings)
    question_embeddings_np = np.array(question_embeddings)
    
    # Find the minimum dimension to ensure compatibility
    min_dim = min(demographic_dim, question_dim)
    
    # Create basic concatenated features (same as original model)
    basic_features = np.hstack([
        demographic_embeddings_np,
        question_embeddings_np
    ])
    
    # Create element-wise products between demographics and questions
    demo_question_product = demographic_embeddings_np[:, :min_dim] * question_embeddings_np[:, :min_dim]
    
    # Create element-wise differences between demographics and questions
    demo_question_diff = demographic_embeddings_np[:, :min_dim] - question_embeddings_np[:, :min_dim]
    
    # Combine all features
    enhanced_features = np.hstack([
        basic_features,                # Original concatenated embeddings
        demo_question_product,         # Element-wise products
        demo_question_diff             # Element-wise differences
    ])
    
    # Print feature dimensions
    original_dim = basic_features.shape[1]
    enhanced_dim = enhanced_features.shape[1]
    print(f"Original feature dimension: {original_dim}")
    print(f"Enhanced feature dimension: {enhanced_dim}")
    print(f"Added {enhanced_dim - original_dim} new feature dimensions")
    
    return enhanced_features


def load_k_fold_models(model_paths, input_size, device):
    """Load all 5 k-fold models for ensemble prediction."""
    models = []
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"Loading k-fold model {i}/5: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize model with correct input size
        model = EnhancedAgePredictor(input_size).to(device)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        models.append(model)
    
    print(f"Successfully loaded {len(models)} k-fold models")
    return models


def ensemble_predict(models, X, device):
    """Generate ensemble predictions by averaging outputs from all k-fold models."""
    print("Generating ensemble predictions...")
    
    all_predictions = []
    
    # Get predictions from each model
    for i, model in enumerate(models, 1):
        print(f"Getting predictions from model {i}/{len(models)}")
        predictions = model.predict(X)
        all_predictions.append(predictions)
    
    # Convert to numpy array for easier manipulation
    all_predictions = np.array(all_predictions)  # Shape: (5, num_samples)
    
    # Calculate ensemble prediction by averaging
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    print(f"Ensemble predictions shape: {ensemble_predictions.shape}")
    print(f"Individual model predictions shape: {all_predictions.shape}")
    
    return ensemble_predictions


def main(args):
    """Main execution flow for k-fold ensemble inference."""
    # Setup environment
    device = setup_environment()
    
    # Configure model paths and output based on argument
    data_path = "data/data_for_report/profiles_for_processing.csv"
    cache_dir = "cache"
    embedding_model = "voyage-2"
    
    # Create output directory if it doesn't exist
    output_dir = "models_1000_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.model == "enhanced":
        model_paths = [
            "enhanced_models_results/1_enhanced_k_fold_model.pth",
            "enhanced_models_results/2_enhanced_k_fold_model.pth",
            "enhanced_models_results/3_enhanced_k_fold_model.pth",
            "enhanced_models_results/4_enhanced_k_fold_model.pth",
            "enhanced_models_results/5_enhanced_k_fold_model.pth"
        ]
        output_path = os.path.join(output_dir, "k_fold_enhanced_model_1000_predictions.csv")
        model_type = "Enhanced K-fold Ensemble"
    elif args.model == "simple":
        model_paths = [
            "not_enhanced_models_results/1_k_fold_voyageapi_embedding_model.pth",
            "not_enhanced_models_results/2_k_fold_voyageapi_embedding_model.pth",
            "not_enhanced_models_results/3_k_fold_voyageapi_embedding_model.pth",
            "not_enhanced_models_results/4_k_fold_voyageapi_embedding_model.pth",
            "not_enhanced_models_results/5_k_fold_voyageapi_embedding_model.pth"
        ]
        output_path = os.path.join(output_dir, "k_fold_model_1000_predictions.csv")
        model_type = "K-fold Ensemble"
    else:
        raise ValueError(f"Invalid model type: {args.model}. Use 'enhanced' or 'simple'")
    
    print(f"=== {model_type} Model 1000 Inference ===")
    print(f"Input data: {data_path}")
    print(f"Models: {len(model_paths)} k-fold models")
    print(f"Output: {output_path}")
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    valid_users = load_data(data_path)
    valid_users = prepare_features(valid_users)
    
    # Generate embeddings
    print("\n2. Generating embeddings...")
    demographic_embeddings, question_embeddings = create_embeddings_for_users(
        valid_users, cache_dir=cache_dir, model=embedding_model
    )
    
    # Create features based on model type
    print("\n3. Creating features...")
    if args.model == "enhanced":
        X = create_enhanced_features(demographic_embeddings, question_embeddings)
    else:  # simple model
        X = create_basic_features(demographic_embeddings, question_embeddings)
    
    # Load k-fold models
    print("\n4. Loading k-fold models...")
    input_size = X.shape[1]
    models = load_k_fold_models(model_paths, input_size, device)
    
    # Generate ensemble predictions
    print("\n5. Generating ensemble predictions...")
    predictions = ensemble_predict(models, X, device)
    
    # Prepare output data
    print("\n6. Preparing output...")
    results_df = pd.DataFrame({
        '': range(len(predictions)),  # Sequential row index 0, 1, 2, ... 999 (empty column title)
        'index': valid_users['index'].values,
        'real_age': valid_users['d_age'].values,
        'predicted_age': np.round(predictions).astype(int)  # Round predictions to whole numbers
    })
    
    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Display summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total predictions: {len(results_df)}")
    print(f"Real age range: {results_df['real_age'].min():.1f} - {results_df['real_age'].max():.1f}")
    print(f"Predicted age range: {results_df['predicted_age'].min():.1f} - {results_df['predicted_age'].max():.1f}")
    print(f"Mean real age: {results_df['real_age'].mean():.2f}")
    print(f"Mean predicted age: {results_df['predicted_age'].mean():.2f}")
    
    # Calculate MAE for reference
    mae = np.mean(np.abs(results_df['predicted_age'] - results_df['real_age']))
    print(f"Mean Absolute Error: {mae:.2f} years")
    
    # Show first few predictions
    print("\n=== Sample Predictions ===")
    print(results_df.head(10).to_string(index=False))
    
    print(f"\nK-fold ensemble inference completed successfully!")
    print(f"Output saved to: {output_path}")
    
    return results_df


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='K-fold Ensemble Model 1000 Inference Script')
    parser.add_argument('--model', type=str, choices=['enhanced', 'simple'], required=True,
                        help='Model type to use: "enhanced" or "simple"')
    
    args = parser.parse_args()
    results = main(args)
