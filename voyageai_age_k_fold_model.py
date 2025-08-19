#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VoyageAI Age Prediction K-Fold Ensemble Model

This script implements an age prediction model for OkCupid users
using VoyageAI embeddings generated from demographic and question response data
with a 5-fold cross-validation ensemble approach.

The model takes demographic and question response features from user profiles, 
creates embeddings, trains 5 separate models using cross-validation, and combines 
their predictions for improved accuracy.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

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
    print(f"Average demographics length: {valid_users['combined_demographics'].apply(len).mean():.0f} characters")
    print(f"Average question responses length: {valid_users['question_responses'].apply(len).mean():.0f} characters")
    
    return valid_users, selected_questions


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
        cache_file=f"{cache_dir}/demographic_embeddings_{user_count}.pkl"
    )
    
    # Generate embeddings for question responses
    question_embeddings = generate_embeddings(
        valid_users['question_responses'].tolist(),
        model=model,
        cache_file=f"{cache_dir}/question_embeddings_{user_count}.pkl"
    )
    
    # Verify embedding dimensions
    print(f"Demographic embeddings shape: {len(demographic_embeddings)} x {len(demographic_embeddings[0])}")
    print(f"Question embeddings shape: {len(question_embeddings)} x {len(question_embeddings[0])}")
    
    # Verify consistency
    assert len(demographic_embeddings) == user_count, f"Demographic embeddings count ({len(demographic_embeddings)}) doesn't match user count ({user_count})"
    assert len(question_embeddings) == user_count, f"Question embeddings count ({len(question_embeddings)}) doesn't match user count ({user_count})"
    
    return demographic_embeddings, question_embeddings


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


def train_model(model, train_loader, val_loader, device, model_save_path, 
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
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
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
    return train_losses, val_losses, model, best_val_loss


def train_k_fold_models(X, y, device, results_dir="results", n_splits=5, 
                        batch_size=32, num_epochs=50, patience=10):
    """Train models using k-fold cross-validation."""
    # Create KFold object
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Track metrics for each fold
    fold_metrics = []
    fold_models = []
    fold_val_predictions = {}  # Store validation predictions for each sample
    
    # Initialize arrays to store validation indices and predictions for all data
    all_val_indices = np.zeros(len(X), dtype=bool)
    all_val_predictions = np.zeros(len(X))
    
    # For each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        fold_num = fold + 1
        print(f"\n{'='*40}\nTraining Fold {fold_num}/{n_splits}\n{'='*40}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create datasets and data loaders
        train_dataset = EmbeddingDataset(X_train, y_train)
        val_dataset = EmbeddingDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_size = X_train.shape[1]
        model = EmbeddingAgePredictor(input_size).to(device)
        
        # Path to save this fold's model
        model_save_path = os.path.join(results_dir, f"{fold_num}_k_fold_voyageapi_embedding_model.pth")
        
        # Train model
        train_losses, val_losses, trained_model, best_val_loss = train_model(
            model, train_loader, val_loader, device,
            model_save_path=model_save_path,
            num_epochs=num_epochs,
            patience=patience
        )
        
        # Evaluate on validation set
        val_predictions = trained_model.predict(X_val)
        val_mae = mae(y_val, val_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        
        # Store metrics
        fold_metrics.append({
            'fold': fold_num,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
        
        # Store model and validation predictions
        fold_models.append(trained_model)
        
        # Store validation indices and predictions
        all_val_indices[val_idx] = True
        all_val_predictions[val_idx] = val_predictions
        
        # Print results
        print(f"\nFold {fold_num} Results:")
        print(f"  Validation MAE: {val_mae:.4f}")
        print(f"  Validation R²: {val_r2:.4f}")
        
    # Calculate overall validation performance using all folds
    overall_val_mae = mae(y[all_val_indices], all_val_predictions[all_val_indices])
    overall_val_r2 = r2_score(y[all_val_indices], all_val_predictions[all_val_indices])
    
    print(f"\nOverall Cross-Validation Results:")
    print(f"  Average Validation MAE: {np.mean([m['val_mae'] for m in fold_metrics]):.4f}")
    print(f"  Average Validation R²: {np.mean([m['val_r2'] for m in fold_metrics]):.4f}")
    print(f"  Combined Validation MAE: {overall_val_mae:.4f}")
    print(f"  Combined Validation R²: {overall_val_r2:.4f}")
    
    # Plot the training history for all folds
    plot_k_fold_training_history(fold_metrics, results_dir)
    
    return fold_models, fold_metrics, overall_val_mae


class EnsemblePredictor:
    """Ensemble predictor that combines multiple models."""
    def __init__(self, models):
        self.models = models
        
    def predict(self, X):
        """Make predictions using the ensemble."""
        # Get predictions from each model
        all_predictions = [model.predict(X) for model in self.models]
        
        # Stack predictions and take average
        stacked_predictions = np.stack(all_predictions)
        ensemble_predictions = np.mean(stacked_predictions, axis=0)
        
        # Clip predictions to reasonable age range (13-100)
        return np.clip(ensemble_predictions, 13, 100)


def calculate_accuracy_metrics(y_true, y_pred, save_path=None, title=None):
    """Calculate age prediction accuracy metrics."""
    # Calculate absolute errors
    abs_errors = np.abs(y_pred - y_true)
    
    # Calculate percentage within different thresholds
    within_1_year = np.mean(abs_errors <= 1) * 100
    within_3_years = np.mean(abs_errors <= 3) * 100
    within_5_years = np.mean(abs_errors <= 5) * 100
    
    # Calculate R² score
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAE
    error = mae(y_true, y_pred)
    
    metrics = {
        'within_1_year': within_1_year,
        'within_3_years': within_3_years, 
        'within_5_years': within_5_years,
        'r2_score': r2,
        'mae': error
    }
    
    header = f"{title} Accuracy Metrics:" if title else "Accuracy Metrics:"
    print(f"\n{header}")
    print(f"  Mean Absolute Error: {error:.2f} years")
    print(f"  Predictions within ±1 year: {within_1_year:.2f}%")
    print(f"  Predictions within ±3 years: {within_3_years:.2f}%")
    print(f"  Predictions within ±5 years: {within_5_years:.2f}%")
    print(f"  R² score: {r2:.4f}")
    
    # Save metrics to file if path is provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(f"{header}\n")
            f.write(f"  Mean Absolute Error: {error:.2f} years\n")
            f.write(f"  Predictions within ±1 year: {within_1_year:.2f}%\n")
            f.write(f"  Predictions within ±3 years: {within_3_years:.2f}%\n")
            f.write(f"  Predictions within ±5 years: {within_5_years:.2f}%\n")
            f.write(f"  R² score: {r2:.4f}\n")
        print(f"Accuracy metrics saved to {save_path}")
    
    return metrics


def create_true_vs_predicted_plot(y_true, y_pred, mean_age, save_path, title=None):
    """Create plot of true ages vs predicted ages with mean age line."""
    # Plot predictions vs actual ages for a sample
    sample_size = min(100, len(y_true))
    sample_indices = np.random.choice(len(y_true), sample_size, replace=False)
    
    sample_y = y_true[sample_indices]
    sample_pred = y_pred[sample_indices]
    
    # Calculate MAE
    error = mae(y_true, y_pred)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    plot_title = title if title else "True vs Predicted Ages"
    fig.suptitle(f"{plot_title} (MAE: {error:.2f} years)", fontsize=16)
    
    # SUBPLOT 1: Line plot showing sorted ages
    # Sort by actual age for better visualization
    sort_idx = np.argsort(sample_y)
    sorted_y = sample_y[sort_idx]
    sorted_pred = sample_pred[sort_idx]
    
    # Create explicit x-axis with integers
    x_indices = np.arange(len(sorted_y))
    
    # Plot actual and predicted ages with explicit integer indices
    ax1.plot(x_indices, sorted_y, 'o-', label='Actual Age')
    ax1.plot(x_indices, sorted_pred, 'x-', label='Predicted Age')
    
    # Add horizontal line for mean age
    ax1.axhline(y=mean_age, color='g', linestyle='--', label=f'Mean Age: {mean_age:.2f}')
    
    ax1.set_title('Sorted by true age')
    ax1.set_xlabel('Sample Index (sorted by actual age)')
    ax1.set_ylabel('Age')
    ax1.grid(True)
    ax1.legend()
    
    # Set x-axis to show integer ticks only
    ax1.set_xticks(x_indices[::5])  # Show every 5th tick to avoid overcrowding
    
    # SUBPLOT 2: Scatter plot of True vs Predicted ages
    ax2.scatter(sample_y, sample_pred, alpha=0.6)
    
    # Add diagonal line for perfect predictions
    min_age = min(min(sample_y), min(sample_pred))
    max_age = max(max(sample_y), max(sample_pred))
    ax2.plot([min_age, max_age], [min_age, max_age], 'r--', label='Perfect Prediction')
    
    # Add horizontal line for mean predictions
    ax2.axhline(y=mean_age, color='g', linestyle='--', label=f'Mean Age: {mean_age:.2f}')
    
    ax2.set_title('Scatter plot')
    ax2.set_xlabel('True Age')
    ax2.set_ylabel('Predicted Age')
    ax2.grid(True)
    ax2.legend()
    
    # Make the plot look nicer
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"True vs Predicted Ages plot saved to {save_path}")
    plt.close()
    
    return error


def plot_k_fold_training_history(fold_metrics, results_dir):
    """Plot the training and validation loss history for all folds."""
    plt.figure(figsize=(12, 8))
    
    # Plot each fold's training history
    for fold_data in fold_metrics:
        fold_num = fold_data['fold']
        train_losses = fold_data['train_losses']
        val_losses = fold_data['val_losses']
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, '--', alpha=0.7, label=f'Fold {fold_num} Train')
        plt.plot(epochs, val_losses, '-', alpha=0.7, label=f'Fold {fold_num} Val')
    
    plt.title('K-Fold Cross-Validation Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(results_dir, "k_fold_training_history.png")
    plt.savefig(save_path)
    print(f"K-fold training history plot saved to {save_path}")
    plt.close()
    
    # Plot ensemble loss vs epoch as a separate visualization
    plot_ensemble_loss_vs_epoch(fold_metrics, results_dir)
    
    return


def plot_ensemble_loss_vs_epoch(fold_metrics, results_dir):
    """
    Plot the average training and validation loss versus epoch.
    This shows how the models perform during training across all folds.
    """
    plt.figure(figsize=(12, 8))
    
    # Find the maximum number of epochs across all folds
    max_epochs = max([len(fold['train_losses']) for fold in fold_metrics])
    epochs = range(1, max_epochs + 1)
    
    # Prepare arrays for averaging
    avg_train_losses = []
    avg_val_losses = []
    
    # Calculate average training and validation losses at each epoch
    for epoch in epochs:
        epoch_train_losses = []
        epoch_val_losses = []
        
        # Collect losses from each fold for this epoch
        for fold_data in fold_metrics:
            train_losses = fold_data['train_losses']
            val_losses = fold_data['val_losses']
            
            if epoch <= len(train_losses):
                epoch_train_losses.append(train_losses[epoch-1])
            
            if epoch <= len(val_losses):
                epoch_val_losses.append(val_losses[epoch-1])
        
        # Calculate average losses for this epoch
        if epoch_train_losses:
            avg_train_losses.append(np.mean(epoch_train_losses))
        
        if epoch_val_losses:
            avg_val_losses.append(np.mean(epoch_val_losses))
    
    # Plot average training loss
    plt.plot(epochs[:len(avg_train_losses)], avg_train_losses, 
             '-o', color='blue', label='Average Training Loss', alpha=0.7)
    
    # Plot average validation loss
    plt.plot(epochs[:len(avg_val_losses)], avg_val_losses, 
             '-o', color='red', label='Average Validation Loss', alpha=0.7)
    
    # Mark best average validation performance
    best_epoch = np.argmin(avg_val_losses) + 1
    best_loss = np.min(avg_val_losses)
    plt.plot(best_epoch, best_loss, 'r*', markersize=12, 
             label=f'Best Validation: {best_loss:.4f} @ epoch {best_epoch}')
    
    # Add labels, title, and legend
    plt.title('Average Training and Validation Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True)
    
    # Add text annotation for best performance
    plt.annotate(f'Best: {best_loss:.4f}',
                xy=(best_epoch, best_loss),
                xytext=(best_epoch + 1, best_loss + 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    # Save figure
    save_path = os.path.join(results_dir, "ensemble_loss_vs_epoch.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Ensemble loss vs epoch plot saved to {save_path}")
    plt.close()
    
    return


def plot_fold_performances(fold_metrics, results_dir):
    """Plot the performance metrics for each fold."""
    folds = [m['fold'] for m in fold_metrics]
    val_maes = [m['val_mae'] for m in fold_metrics]
    val_r2s = [m['val_r2'] * 100 for m in fold_metrics]  # Scale R² to percentage for better visibility
    
    # Calculate statistics
    mean_mae = np.mean(val_maes)
    std_mae = np.std(val_maes)
    mean_r2 = np.mean(val_r2s) / 100  # Convert back to original scale for display
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot MAE bars
    x = np.arange(len(folds))
    width = 0.35
    rects1 = ax1.bar(x - width/2, val_maes, width, label='MAE (years)', color='tab:blue', alpha=0.7)
    
    # Add a second y-axis for R²
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, val_r2s, width, label='R² (%)', color='tab:orange', alpha=0.7)
    
    # Add labels, title, and legend
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('MAE (years)')
    ax2.set_ylabel('R² (%)')
    ax1.set_title(f'Performance by Fold (Avg MAE: {mean_mae:.2f} ± {std_mae:.2f}, Avg R²: {mean_r2:.4f})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add horizontal line for mean MAE
    ax1.axhline(y=mean_mae, color='tab:blue', linestyle='--', alpha=0.5)
    
    # Add value labels to the bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(
            f'{height:.2f}',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom'
        )
    
    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(
            f'{height:.1f}',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom'
        )
    
    # Save figure
    save_path = os.path.join(results_dir, "fold_performance_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Fold performance comparison plot saved to {save_path}")
    plt.close()
    
    return

def evaluate_individual_and_ensemble_models(fold_models, X_test, y_test, results_dir):
    """
    Evaluate individual fold models and their ensemble on the test set.
    
    Returns:
        ensemble_metrics: Dictionary of ensemble metrics
        individual_metrics: List of dictionaries with individual model metrics
    """
    # Create ensemble
    ensemble = EnsemblePredictor(fold_models)
    
    # Mean age for plots
    mean_age = np.mean(y_test)
    
    # Evaluate ensemble model
    ensemble_predictions = ensemble.predict(X_test)
    ensemble_error = mae(y_test, ensemble_predictions)
    ensemble_r2 = r2_score(y_test, ensemble_predictions)
    
    print(f"\nEnsemble Model Performance:")
    print(f"  Test MAE: {ensemble_error:.4f}")
    print(f"  Test R²: {ensemble_r2:.4f}")
    
    # Save ensemble metrics and plot
    ensemble_metrics_path = os.path.join(results_dir, "k_fold_ensemble_accuracy.txt")
    ensemble_plot_path = os.path.join(results_dir, "k_fold_ensemble_TrueAgeVSPredictedAge.png")
    
    # Calculate and save detailed metrics
    ensemble_metrics = calculate_accuracy_metrics(
        y_test, ensemble_predictions, 
        save_path=ensemble_metrics_path,
        title="K-Fold Ensemble"
    )
    
    # Create and save plot
    create_true_vs_predicted_plot(
        y_test, ensemble_predictions, mean_age, 
        save_path=ensemble_plot_path,
        title="K-Fold Ensemble"
    )
    
    # Evaluate individual models
    individual_metrics = []
    for i, model in enumerate(fold_models):
        fold_num = i + 1
        predictions = model.predict(X_test)
        fold_error = mae(y_test, predictions)
        fold_r2 = r2_score(y_test, predictions)
        
        individual_metrics.append({
            'fold': fold_num,
            'test_mae': fold_error,
            'test_r2': fold_r2
        })
        
        print(f"Fold {fold_num} Model Test Performance:")
        print(f"  Test MAE: {fold_error:.4f}")
        print(f"  Test R²: {fold_r2:.4f}")
    
    # Plot individual vs ensemble performance
    plot_individual_vs_ensemble(individual_metrics, ensemble_error, results_dir)
    
    return ensemble_metrics, individual_metrics


def plot_individual_vs_ensemble(individual_metrics, ensemble_error, results_dir):
    """Plot comparison of individual models vs ensemble performance."""
    folds = [m['fold'] for m in individual_metrics]
    test_maes = [m['test_mae'] for m in individual_metrics]
    
    # Calculate statistics
    mean_indiv_mae = np.mean(test_maes)
    
    plt.figure(figsize=(10, 6))
    
    # Plot individual model bars
    x = np.arange(len(folds) + 1)  # +1 for ensemble
    plt.bar(x[:-1], test_maes, alpha=0.7, label='Individual Models')
    
    # Plot ensemble bar with different color
    plt.bar(x[-1], ensemble_error, color='green', alpha=0.7, label='Ensemble')
    
    # Add horizontal line for mean of individual models
    plt.axhline(y=mean_indiv_mae, color='blue', linestyle='--', alpha=0.5, 
                label=f'Avg Individual (MAE: {mean_indiv_mae:.2f})')
    
    # Add horizontal line for ensemble
    plt.axhline(y=ensemble_error, color='green', linestyle='--', alpha=0.5,
                label=f'Ensemble (MAE: {ensemble_error:.2f})')
    
    # Calculate improvement
    improvement = (mean_indiv_mae - ensemble_error) / mean_indiv_mae * 100
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Test MAE (years)')
    plt.title(f'Individual Models vs Ensemble Performance\nEnsemble improves by {improvement:.2f}%')
    plt.xticks(x, folds + ['Ensemble'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, mae_value in enumerate(test_maes + [ensemble_error]):
        plt.annotate(
            f'{mae_value:.2f}',
            xy=(i, mae_value),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom'
        )
    
    # Save figure
    save_path = os.path.join(results_dir, "individual_vs_ensemble_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Individual vs Ensemble comparison plot saved to {save_path}")
    plt.close()
    
    return improvement


def plot_error_distribution(y_true, y_pred, results_dir, title=None):
    """Plot the distribution of prediction errors."""
    errors = y_pred - y_true
    
    plt.figure(figsize=(12, 6))
    
    # Create histogram with density plot
    sns.histplot(errors, kde=True)
    
    # Add vertical line at 0 (perfect prediction)
    plt.axvline(x=0, color='r', linestyle='--', label='Perfect Prediction')
    
    # Add mean error line
    mean_error = np.mean(errors)
    plt.axvline(x=mean_error, color='g', linestyle='--', 
                label=f'Mean Error: {mean_error:.2f}')
    
    # Calculate metrics
    std_error = np.std(errors)
    abs_errors = np.abs(errors)
    median_abs_error = np.median(abs_errors)
    
    # Add labels and title
    plt_title = title if title else "Age Prediction Error Distribution"
    plt.title(f"{plt_title}\nMean Error: {mean_error:.2f}, Std Dev: {std_error:.2f}, Median Abs Error: {median_abs_error:.2f}")
    plt.xlabel('Prediction Error (years)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plot_name = "k_fold_ensemble_error_distribution.png" if "Ensemble" in (title or "") else "error_distribution.png"
    save_path = os.path.join(results_dir, plot_name)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Error distribution plot saved to {save_path}")
    plt.close()
    
    return mean_error, std_error, median_abs_error


def main(args):
    """Main execution flow for k-fold ensemble approach."""
    # Setup environment
    device = setup_environment()
    
    # Create fixed results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to {results_dir}")
    
    # Load and prepare data
    csv_data, question_data = load_data(args.data_path, args.question_data_path)
    valid_users, selected_questions = prepare_features(csv_data, question_data, args.sample_size)
    
    # Generate embeddings
    demographic_embeddings, question_embeddings = create_embeddings_for_users(
        valid_users, cache_dir=args.cache_dir, model=args.embedding_model
    )
    
    # Combine embeddings into feature matrix
    X = np.hstack([
        demographic_embeddings,
        question_embeddings
    ])
    
    # Target variable: age
    y = valid_users['d_age'].values
    
    # Split into training/cross-validation set and hold-out test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training/validation data shape: {X_train_val.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Train k-fold models
    print(f"\n{'='*60}\nTraining 5-Fold Cross-Validation Models\n{'='*60}")
    fold_models, fold_metrics, overall_val_mae = train_k_fold_models(
        X_train_val, y_train_val, device, 
        results_dir=results_dir, 
        n_splits=5,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience
    )
    
    # Plot fold performances
    plot_fold_performances(fold_metrics, results_dir)
    
    # Evaluate individual models and ensemble on test set
    print(f"\n{'='*60}\nEvaluating Models on Test Set\n{'='*60}")
    ensemble_metrics, individual_metrics = evaluate_individual_and_ensemble_models(
        fold_models, X_test, y_test, results_dir
    )
    
    # Compare with baseline (mean prediction)
    class MeanBaseline:
        def __init__(self, y_train):
            self.mean_age = np.mean(y_train)
        
        def predict(self, X):
            return np.full(len(X), self.mean_age)
    
    baseline = MeanBaseline(y_train_val)
    baseline_error = mae(y_test, baseline.predict(X_test))
    print(f"\nMean Baseline MAE: {baseline_error:.4f}")
    
    # Calculate overall improvement over baseline
    ensemble_error = ensemble_metrics['mae']
    improvement = (baseline_error - ensemble_error) / baseline_error * 100
    print(f"K-Fold ensemble improves over the baseline by {improvement:.2f}%")
    
    # Plot error distribution
    ensemble = EnsemblePredictor(fold_models)
    ensemble_predictions = ensemble.predict(X_test)
    plot_error_distribution(y_test, ensemble_predictions, results_dir, title="K-Fold Ensemble")
    
    # Save summary
    summary_path = os.path.join(results_dir, "k_fold_ensemble_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"K-Fold Ensemble Age Prediction Model Summary\n")
        f.write(f"===============================================\n\n")
        f.write(f"Dataset Size: {len(valid_users)} users\n")
        f.write(f"Number of Folds: 5\n")
        f.write(f"Test Set Size: {len(X_test)} users\n\n")
        
        f.write(f"Ensemble Test MAE: {ensemble_error:.2f} years\n")
        f.write(f"Baseline Test MAE: {baseline_error:.2f} years\n")
        f.write(f"Improvement: {improvement:.2f}%\n\n")
        
        f.write(f"Individual Fold Test MAE:\n")
        for metric in individual_metrics:
            f.write(f"  Fold {metric['fold']}: {metric['test_mae']:.2f} years\n")
    
    print(f"\nAll results have been saved to {results_dir}")
    print(f"Summary saved to {summary_path}")
    
    return {
        'ensemble_error': ensemble_error,
        'baseline_error': baseline_error,
        'improvement': improvement,
        'results_dir': results_dir
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='VoyageAI Age Prediction K-Fold Ensemble Model')
    
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
    
    args = parser.parse_args()
    
    # Run the main function
    results = main(args)
    print("Execution completed successfully!")
