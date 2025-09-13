#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loader for LLM Age Prediction

This module provides functionality to load and filter user data for age prediction.
Handles loading user responses and their ages, with column selection capabilities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import os

class UserDataLoader:
    """
    Loads and processes user data for LLM-based age prediction.
    
    This class handles loading user responses to questions along with their demographic
    information, particularly age, which is used as the target variable.
    """
    
    def __init__(self, 
                 user_data_path: str = "../data/user_data_public.csv",
                 question_data_path: str = "../data/question_data.csv"):
        """
        Initialize the data loader.
        
        Args:
            user_data_path: Path to the user data CSV file
            question_data_path: Path to the question metadata CSV file
        """
        self.user_data_path = user_data_path
        self.question_data_path = question_data_path
        self.user_data = None
        self.question_metadata = None
        
    def load_data(self, force_reload: bool = False) -> None:
        """
        Load the user data and question metadata from CSV files.
        
        Args:
            force_reload: If True, reload data even if already loaded
        """
        if self.user_data is None or force_reload:
            print(f"Loading user data from {self.user_data_path}...")
            self.user_data = pd.read_csv(self.user_data_path, low_memory=False)
            print(f"Loaded {len(self.user_data)} user records")
            
        if self.question_metadata is None or force_reload:
            print(f"Loading question metadata from {self.question_data_path}...")
            self.question_metadata = pd.read_csv(self.question_data_path, sep=';')
            # Create a lookup dict for quick access
            self._question_lookup = self.question_metadata.set_index('question').to_dict('index')
            print(f"Loaded metadata for {len(self.question_metadata)} questions")
    
    def get_rows(self, 
                 columns: Optional[List[str]] = None,
                 n_samples: Optional[int] = None,
                 min_age: Optional[int] = None,
                 max_age: Optional[int] = None,
                 random_seed: Optional[int] = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get user rows with optional filtering and column selection.
        
        Args:
            columns: List of column names to include. If None, includes all question columns
            n_samples: Number of samples to return. If None, returns all matching rows
            min_age: Minimum age to include (inclusive)
            max_age: Maximum age to include (inclusive)
            random_seed: Random seed for sampling reproducibility
            
        Returns:
            Tuple of (user_data_df, ages_series)
            - user_data_df: DataFrame with selected columns and filtered rows
            - ages_series: Series with corresponding ages
        """
        if self.user_data is None:
            self.load_data()
            
        # Start with all data
        data = self.user_data.copy()
        
        # Filter by age if specified
        age_col = 'd_age'
        if age_col not in data.columns:
            raise ValueError(f"Age column '{age_col}' not found in data")
            
        # Remove rows with missing age
        initial_count = len(data)
        data = data[data[age_col].notna()]
        print(f"Filtered out {initial_count - len(data)} rows with missing age")
        
        if min_age is not None:
            data = data[data[age_col] >= min_age]
            print(f"Applied minimum age filter: {min_age}")
            
        if max_age is not None:
            data = data[data[age_col] <= max_age]
            print(f"Applied maximum age filter: {max_age}")
        
        # Select columns if specified
        if columns is None:
            # Default to all question columns plus demographics
            question_columns = [col for col in data.columns if col.startswith('q')]
            demographic_columns = [col for col in data.columns if col.startswith('d_') or col.startswith('p_')]
            columns = question_columns + demographic_columns
        else:
            # Validate that requested columns exist
            missing_cols = [col for col in columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
        
        # Always include the age column for return
        if age_col not in columns:
            columns_to_select = columns + [age_col]
        else:
            columns_to_select = columns
            
        selected_data = data[columns_to_select]
        
        # Sample if requested
        if n_samples is not None and n_samples < len(selected_data):
            if random_seed is not None:
                np.random.seed(random_seed)
            selected_data = selected_data.sample(n=n_samples, random_state=random_seed)
            print(f"Sampled {n_samples} rows")
        
        # Separate features and target
        ages = selected_data[age_col]
        features = selected_data.drop(columns=[age_col] if age_col in selected_data.columns else [])
        
        print(f"Returning {len(features)} rows with {len(features.columns)} feature columns")
        
        return features, ages
    
    def get_question_info(self, question_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific question.
        
        Args:
            question_id: The question ID (e.g., 'q2', 'q11')
            
        Returns:
            Dictionary with question metadata or None if not found
        """
        if self.question_metadata is None:
            self.load_data()
            
        return self._question_lookup.get(question_id)
    
    def get_available_columns(self, column_type: str = "all") -> List[str]:
        """
        Get list of available columns in the dataset.
        
        Args:
            column_type: Filter by type - 'questions', 'demographics', 'personality', 'all'
            
        Returns:
            List of column names
        """
        if self.user_data is None:
            self.load_data()
            
        all_columns = list(self.user_data.columns)
        
        if column_type == "questions":
            return [col for col in all_columns if col.startswith('q')]
        elif column_type == "demographics":
            return [col for col in all_columns if col.startswith('d_')]
        elif column_type == "personality":
            return [col for col in all_columns if col.startswith('p_')]
        else:
            return all_columns
    
    def get_data_summary(self) -> Dict:
        """
        Get a summary of the loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        if self.user_data is None:
            self.load_data()
            
        age_col = 'd_age'
        valid_ages = self.user_data[age_col].dropna()
        
        return {
            'total_rows': len(self.user_data),
            'total_columns': len(self.user_data.columns),
            'question_columns': len([col for col in self.user_data.columns if col.startswith('q')]),
            'demographic_columns': len([col for col in self.user_data.columns if col.startswith('d_')]),
            'personality_columns': len([col for col in self.user_data.columns if col.startswith('p_')]),
            'age_stats': {
                'count': len(valid_ages),
                'mean': valid_ages.mean(),
                'std': valid_ages.std(),
                'min': valid_ages.min(),
                'max': valid_ages.max(),
                'median': valid_ages.median()
            }
        }

def main():
    """Example usage of the UserDataLoader class."""
    # Initialize loader
    loader = UserDataLoader()
    
    # Print data summary
    summary = loader.get_data_summary()
    print("Data Summary:")
    print(f"Total rows: {summary['total_rows']}")
    print(f"Question columns: {summary['question_columns']}")
    print(f"Age stats: mean={summary['age_stats']['mean']:.1f}, "
          f"range=[{summary['age_stats']['min']}-{summary['age_stats']['max']}]")
    
    # Get a sample of data
    print("\nGetting sample data...")
    features, ages = loader.get_rows(
        n_samples=10,
        min_age=18,
        max_age=65
    )
    
    print(f"Sample features shape: {features.shape}")
    print(f"Sample ages: {list(ages)}")
    print(f"Sample question columns: {[col for col in features.columns if col.startswith('q')][:5]}...")

if __name__ == "__main__":
    main()
