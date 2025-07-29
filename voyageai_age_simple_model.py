#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VoyageAI Age Prediction Simple Model

This script implements an age prediction model for OkCupid users
using VoyageAI embeddings generated from text data.

The model takes text features from user profiles and predicts their age.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
from datetime import datetime

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Visualization
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # Use non-interactive backend
import seaborn as sns

# VoyageAI API
import voyageai


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


def load_data(data_path, question_data_path, sample_size=None):
    """Load and prepare the OkCupid dataset."""
    # Load datasets
    print(f"Loading dataset from {data_path}")
    csv = pd.read_csv(data_path, low_memory=False)
    
    print(f"Loading question data from {question_data_path}")
    question_csv = pd.read_csv(question_data_path, delimiter=';', low_memory=False, index_col=0)
    
    # Basic exploration
    print(f"Dataset shape: {csv.shape}")
    print(f"Number of questions: {len([col for col in csv.columns if col.startswith('q')])}")
    
    # Check target variable
    print(f"Age statistics:")
    print(f"Min age: {csv['d_age'].min()}")
    print(f"Max age: {csv['d_age'].max()}")
    print(f"Mean age: {csv['d_age'].mean():.2f}")
    print(f"Median age: {csv['d_age'].median():.2f}")
    
    return csv, question_csv


def prepare_features(csv, question_csv, sample_size=None):
    """Extract and prepare text features from the dataset."""
    # Get essay columns (free-form text fields)
    essay_columns = [col for col in csv.columns if col.startswith('essay')]
    print(f"Found {len(essay_columns)} essay columns")
    
    # Define possible demographic columns
    all_demographic_columns = [
        'd_body_type', 'd_diet', 'd_drinks', 'd_drugs', 'd_education', 
        'd_ethnicity', 'd_income', 'd_job', 'd_orientation', 
        'd_pets', 'd_religion', 'd_religion_seriosity', 'd_sign', 
        'd_smokes', 'd_speaks', 'speaks', 'status'
    ]
    
    # Filter to only include columns that exist in the dataset
    demographic_columns = [col for col in all_demographic_columns if col in csv.columns]
    print(f"Found {len(demographic_columns)} demographic columns: {demographic_columns}")
    
    # Get high-response questions - using index which contains question IDs
    # The N column contains the number of responses
    high_response_questions = question_csv[question_csv['N'] > 50000].index.tolist()
    
    # Process users with non-null age
    valid_users = csv.dropna(subset=['d_age'])
    print(f"Users with valid age: {len(valid_users)}")
    
    # Sample a subset if specified
    if sample_size:
        valid_users = valid_users.sample(sample_size, random_state=42)
    print(f"Working with {len(valid_users)} users")
    
    # Combine essays into a single text field
    if essay_columns:
        valid_users['combined_essays'] = valid_users[essay_columns].fillna("").astype(str).agg(' '.join, axis=1).apply(clean_text)
    else:
        print("Warning: No essay columns found. Using empty strings.")
        valid_users['combined_essays'] = ""
    
    # Combine demographic information
    if demographic_columns:
        valid_users['combined_demographics'] = valid_users[demographic_columns].fillna("unknown").astype(str).agg(' '.join, axis=1).apply(clean_text)
    else:
        print("Warning: No demographic columns found. Using empty strings.")
        valid_users['combined_demographics'] = ""
    
    # Extract question responses (focusing on a subset of high-response questions)
    # We'll take the top 20 questions for this example
    selected_questions = high_response_questions[:20] if len(high_response_questions) > 20 else high_response_questions
    
    # Check if all selected questions are in the dataset columns
    valid_questions = [q for q in selected_questions if q in csv.columns]
    if len(valid_questions) < len(selected_questions):
        print(f"Warning: {len(selected_questions) - len(valid_questions)} questions not found in dataset")
    selected_questions = valid_questions
    
    # Create combined text from question responses
    if selected_questions:
        valid_users['question_responses'] = valid_users[selected_questions].fillna("no_answer").astype(str).agg(' '.join, axis=1).apply(clean_text)
    else:
        print("Warning: No valid questions found. Using empty strings.")
        valid_users['question_responses'] = ""
    
    # Display statistics about our text features
    print(f"Average essay length: {valid_users['combined_essays'].apply(len).mean():.0f} characters")
    print(f"Average demographics length: {valid_users['combined_demographics'].apply(len).mean():.0f} characters")
    print(f"Average question responses length: {valid_users['question_responses'].apply(len).mean():.0f} characters")
    
    return valid_users, selected_questions


def generate_embeddings(texts, model="voyage-2", batch_size=20, cache_file=None):
    """Generate embeddings for a list of texts using VoyageAI API or random vectors for testing."""
    # Check if cache exists and load it
    if cache_file and os.path.exists(cache_file):
        print(f"Loading embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Set embedding size based on model
    embedding_size = 1024 if model == "voyage-2" else 768
    
    # Check if API key is set
    api_key = os.getenv("VOYAGE_API_KEY")
    if api_key and api_key != "your_api_key_here":
        # Use the actual API if key is provided
        all_embeddings = []
        
        # Process in batches to handle API rate limits
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings with API"):
            batch = texts[i:i + batch_size]
            try:
                # Remove empty strings to avoid API errors
                batch = [text if text.strip() else "unknown" for text in batch]
                
                # Call VoyageAI API
                response = voyageai.embed(texts=batch, model=model)
                batch_embeddings = response.embeddings
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error in batch {i}-{i+batch_size}: {e}")
                # Handle error by adding zero embeddings
                zero_embeddings = [[0.0] * embedding_size for _ in range(len(batch))]
                all_embeddings.extend(zero_embeddings)
    else:
        # Generate random embeddings for testing if no API key
        print(f"No valid API key found. Generating random embeddings for testing...")
        np.random.seed(42)  # For reproducibility
        all_embeddings = [np.random.normal(0, 1, embedding_size).tolist() for _ in range(len(texts))]
    
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
    
    # Generate embeddings for essays
    essay_embeddings = generate_embeddings(
        valid_users['combined_essays'].tolist(),
        model=model,
        cache_file=f"{cache_dir}/essay_embeddings.pkl"
    )
    
    # Generate embeddings for demographics
    demographic_embeddings = generate_embeddings(
        valid_users['combined_demographics'].tolist(),
        model=model,
        cache_file=f"{cache_dir}/demographic_embeddings.pkl"
    )
    
    # Generate embeddings for question responses
    question_embeddings = generate_embeddings(
        valid_users['question_responses'].tolist(),
        model=model,
        cache_file=f"{cache_dir}/question_embeddings.pkl"
    )
    
    # Verify embedding dimensions
    print(f"Essay embeddings shape: {len(essay_embeddings)} x {len(essay_embeddings[0])}")
    print(f"Demographic embeddings shape: {len(demographic_embeddings)} x {len(demographic_embeddings[0])}")
    print(f"Question embeddings shape: {len(question_embeddings)} x {len(question_embeddings[0])}")
    
    return essay_embeddings, demographic_embeddings, question_embeddings


class EmbeddingDataset(Dataset):
    """PyTorch Dataset for embeddings and targets."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EmbeddingAgePredictor(nn.Module):
    """Neural network for age prediction using embeddings."""
    def __init__(self, input_size):
        super(EmbeddingAgePredictor, self).__init__()
        
        # Model architecture
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
        # Initialize weights for better training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
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


def train_model(model, train_loader, test_loader, device, model_save_path="best_model.pth", 
                num_epochs=50, patience=10):
    """Train the model with early stopping."""
    # Define loss function and optimizer
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training parameters
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
            print(f"  Saved new best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  Early stopping patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    
    # Return training history for plotting
    return train_losses, val_losses, model


def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot the training and validation loss history."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    # Only show plots interactively when not running in script mode
    # plt.show()
    plt.close()


def evaluate_model(model, X, y, name="Model", save_dir=None):
    """Evaluate model performance with visualizations."""
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate MAE
    error = mae(y, predictions)
    print(f"{name} MAE: {error:.2f} years")
    
    # Plot predictions vs actual ages for a sample
    sample_size = min(100, len(y))
    sample_indices = np.random.choice(len(y), sample_size, replace=False)
    
    sample_y = y[sample_indices]
    sample_pred = predictions[sample_indices]
    
    plt.figure(figsize=(12, 6))
    
    # Sort by actual age for better visualization
    sort_idx = np.argsort(sample_y)
    sample_y = sample_y[sort_idx]
    sample_pred = sample_pred[sort_idx]
    
    plt.plot(sample_y, 'o-', label='Actual Age')
    plt.plot(sample_pred, 'x-', label='Predicted Age')
    plt.title(f'{name} Predictions vs Actual Ages (MAE: {error:.2f})')
    plt.xlabel('Sample Index (sorted by actual age)')
    plt.ylabel('Age')
    plt.grid(True)
    plt.legend()
    
    if save_dir:
        plot_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_predictions.png")
        plt.savefig(plot_path)
        print(f"Predictions plot saved to {plot_path}")
    
    # plt.show()
    plt.close()
    
    # Plot error distribution
    errors = predictions - y
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(f'{name} Error Distribution')
    plt.xlabel('Prediction Error (years)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.axvline(0, color='red', linestyle='--')
    
    if save_dir:
        error_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_error_dist.png")
        plt.savefig(error_path)
        print(f"Error distribution plot saved to {error_path}")
    
    # plt.show()
    plt.close()
    
    return error


def analyze_feature_importance(essay_embeddings, demographic_embeddings, question_embeddings, 
                              X_train, y_train, X_test, y_test, device, save_dir=None):
    """Analyze which embedding types contribute most to predictions."""
    
    # Function to evaluate model with only certain embedding types
    def evaluate_partial_embeddings(embedding_type):
        """Train and evaluate a model with only certain types of embeddings."""
        if embedding_type == 'essays':
            X_partial = np.array(essay_embeddings)
        elif embedding_type == 'demographics':
            X_partial = np.array(demographic_embeddings)
        elif embedding_type == 'questions':
            X_partial = np.array(question_embeddings)
        elif embedding_type == 'essays+demographics':
            X_partial = np.hstack([essay_embeddings, demographic_embeddings])
        elif embedding_type == 'essays+questions':
            X_partial = np.hstack([essay_embeddings, question_embeddings])
        elif embedding_type == 'demographics+questions':
            X_partial = np.hstack([demographic_embeddings, question_embeddings])
        elif embedding_type == 'all':
            X_partial = np.hstack([essay_embeddings, demographic_embeddings, question_embeddings])
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        # Split into train and test
        X_partial_train, X_partial_test = X_partial[:len(X_train)], X_partial[len(X_train):]
        
        # Create and train a model
        partial_model = EmbeddingAgePredictor(X_partial_train.shape[1]).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(partial_model.parameters(), lr=0.001)
        
        # Create datasets and loaders
        train_dataset = EmbeddingDataset(X_partial_train, y_train)
        test_dataset = EmbeddingDataset(X_partial_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Simple training loop (fewer epochs for this analysis)
        for epoch in range(10):
            partial_model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = partial_model(inputs).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        error = mae(y_test, partial_model.predict(X_partial_test))
        return error

    # Analyze contribution of different embedding types
    embedding_types = [
        'essays', 'demographics', 'questions', 
        'essays+demographics', 'essays+questions', 'demographics+questions', 'all'
    ]
    
    # Record results in a dictionary
    results = {}
    
    print("Feature importance analysis:")
    for emb_type in embedding_types:
        error = evaluate_partial_embeddings(emb_type)
        results[emb_type] = error
        print(f"  {emb_type}: MAE = {error:.2f} years")

    # Convert to DataFrame for better visualization
    results_df = pd.DataFrame(list(results.items()), columns=['Features', 'MAE'])
    results_df = results_df.sort_values('MAE')
    
    # Plot results
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Features', y='MAE', data=results_df, palette='viridis')
    plt.title('Feature Importance: MAE for Different Feature Combinations')
    plt.xlabel('Feature Set')
    plt.ylabel('Mean Absolute Error (years)')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_dir:
        importance_path = os.path.join(save_dir, "feature_importance.png")
        plt.savefig(importance_path)
        print(f"Feature importance plot saved to {importance_path}")
    
    # plt.show()
    plt.close()
    
    print(f"Best feature set: {results_df.iloc[0]['Features']} with MAE: {results_df.iloc[0]['MAE']:.2f} years")
    
    return results_df


def main(args):
    """Main execution flow."""
    # Setup environment
    device = setup_environment()
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")
    
    # Load and prepare data
    csv_data, question_data = load_data(args.data_path, args.question_data_path)
    valid_users, selected_questions = prepare_features(csv_data, question_data, args.sample_size)
    
    # Generate embeddings
    essay_embeddings, demographic_embeddings, question_embeddings = create_embeddings_for_users(
        valid_users, cache_dir=args.cache_dir, model=args.embedding_model
    )
    
    # Combine embeddings into feature matrix
    X = np.hstack([
        essay_embeddings,
        demographic_embeddings,
        question_embeddings
    ])
    
    # Target variable: age
    y = valid_users['d_age'].values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Create datasets and dataloaders
    train_dataset = EmbeddingDataset(X_train, y_train)
    test_dataset = EmbeddingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = EmbeddingAgePredictor(input_size).to(device)
    
    # Train model
    model_save_path = os.path.join(output_dir, "best_embedding_age_model.pth")
    train_losses, val_losses, trained_model = train_model(
        model, train_loader, test_loader, device, 
        model_save_path=model_save_path,
        num_epochs=args.epochs,
        patience=args.patience
    )
    
    # Plot training history
    history_plot_path = os.path.join(output_dir, "training_history.png")
    plot_training_history(train_losses, val_losses, save_path=history_plot_path)
    
    # Evaluate model
    model_error = evaluate_model(trained_model, X_test, y_test, "VoyageAPI Embedding Model", save_dir=output_dir)
    
    # Compare with baseline (mean prediction)
    class MeanBaseline:
        def __init__(self, y_train):
            self.mean_age = np.mean(y_train)
        
        def predict(self, X):
            return np.full(len(X), self.mean_age)
    
    baseline = MeanBaseline(y_train)
    baseline_error = evaluate_model(baseline, X_test, y_test, "Mean Baseline", save_dir=output_dir)
    
    # Calculate improvement
    improvement = (baseline_error - model_error) / baseline_error * 100
    print(f"Our model improves over the baseline by {improvement:.2f}%")
    
    # Feature importance analysis
    if args.feature_analysis:
        print("\nAnalyzing feature importance...")
        results_df = analyze_feature_importance(
            essay_embeddings, demographic_embeddings, question_embeddings,
            X_train, y_train, X_test, y_test, device, save_dir=output_dir
        )
        
        # Save feature importance results
        results_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    
    print(f"\nAll results have been saved to {output_dir}")
    
    return {
        'model_error': model_error,
        'baseline_error': baseline_error,
        'improvement': improvement,
        'model_path': model_save_path,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='VoyageAI Age Prediction Simple Model')
    
    parser.add_argument('--data_path', type=str, default='data/user_data_public.csv',
                        help='Path to the OkCupid user data CSV file')
    parser.add_argument('--question_data_path', type=str, default='data/question_data.csv',
                        help='Path to the OkCupid question data CSV file')
    parser.add_argument('--cache_dir', type=str, default='cache',
                        help='Directory to cache embeddings')
    parser.add_argument('--embedding_model', type=str, default='voyage-2',
                        help='VoyageAI model to use for embeddings')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of users to sample (None for all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--feature_analysis', action='store_true',
                        help='Perform feature importance analysis')
    
    args = parser.parse_args()
    
    # Run the main function
    results = main(args)
    print("Execution completed successfully!")
