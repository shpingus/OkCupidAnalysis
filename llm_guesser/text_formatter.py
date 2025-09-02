#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Formatter for LLM Age Prediction

This module converts user data rows into descriptive text blocks that can be
fed to LLMs for age prediction. It uses actual question text, answer options,
and keywords instead of raw question IDs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .data_loader import UserDataLoader

class TextFormatter:
    """
    Converts user response data into descriptive text for LLM consumption.
    
    This class takes raw user responses (like q2=1, q11=3) and converts them into
    natural language descriptions using the actual question text and answer options.
    """
    
    def __init__(self, data_loader: UserDataLoader):
        """
        Initialize the text formatter.
        
        Args:
            data_loader: Initialized UserDataLoader instance
        """
        self.data_loader = data_loader
        # Ensure data is loaded
        if self.data_loader.user_data is None or self.data_loader.question_metadata is None:
            self.data_loader.load_data()
    
    def format_user_row(self, 
                       user_row: pd.Series, 
                       max_questions: Optional[int] = 50,
                       include_demographics: bool = True,
                       include_keywords: bool = True,
                       verbose: bool = False) -> str:
        """
        Convert a user data row into descriptive text.
        
        Args:
            user_row: Series containing user responses
            max_questions: Maximum number of questions to include (None for all)
            include_demographics: Whether to include demographic information
            include_keywords: Whether to include question keywords/categories
            verbose: Whether to include question IDs for debugging
            
        Returns:
            Formatted text description of the user
        """
        text_parts = []
        
        # Process question responses
        question_responses = self._format_question_responses(
            user_row, max_questions, include_keywords, verbose
        )
        if question_responses:
            text_parts.append("**Question Responses:**")
            text_parts.append(question_responses)
        
        # Process demographic information
        if include_demographics:
            demographic_info = self._format_demographics(user_row)
            if demographic_info:
                text_parts.append("**Demographics:**")
                text_parts.append(demographic_info)
        
        return "\n\n".join(text_parts) if text_parts else "No data available"
    
    def _format_question_responses(self, 
                                  user_row: pd.Series, 
                                  max_questions: Optional[int],
                                  include_keywords: bool,
                                  verbose: bool) -> str:
        """
        Format question responses into natural language.
        
        Args:
            user_row: User data row
            max_questions: Maximum questions to process
            include_keywords: Include question categories/keywords
            verbose: Include question IDs
            
        Returns:
            Formatted question responses text
        """
        formatted_responses = []
        question_columns = [col for col in user_row.index if col.startswith('q')]
        
        # Limit number of questions if specified
        if max_questions is not None:
            # Prioritize questions with non-null responses
            non_null_questions = [col for col in question_columns if pd.notna(user_row[col])]
            null_questions = [col for col in question_columns if pd.isna(user_row[col])]
            
            # Take non-null first, then null up to max_questions
            prioritized_questions = non_null_questions[:max_questions]
            if len(prioritized_questions) < max_questions:
                remaining_slots = max_questions - len(prioritized_questions)
                prioritized_questions.extend(null_questions[:remaining_slots])
            
            question_columns = prioritized_questions
        
        for question_id in question_columns:
            response_text = self._format_single_response(
                question_id, user_row[question_id], include_keywords, verbose
            )
            if response_text:
                formatted_responses.append(response_text)
        
        return "\n".join(formatted_responses)
    
    def _format_single_response(self, 
                               question_id: str, 
                               response_value: any,
                               include_keywords: bool,
                               verbose: bool) -> Optional[str]:
        """
        Format a single question response.
        
        Args:
            question_id: Question ID (e.g., 'q2')
            response_value: User's response to the question
            include_keywords: Include question keywords
            verbose: Include question ID
            
        Returns:
            Formatted response text or None if question not found
        """
        question_info = self.data_loader.get_question_info(question_id)
        if not question_info:
            return None
        
        question_text = question_info.get('text', 'Unknown question')
        keywords = question_info.get('Keywords', '')
        
        # Handle missing/null responses
        if pd.isna(response_value):
            response_text = "[No response]"
        else:
            # Try to map response to option text
            response_text = self._map_response_to_text(question_info, response_value)
        
        # Build the formatted response
        parts = []
        
        # Add question text
        parts.append(f"Q: {question_text}")
        
        # Add response
        parts.append(f"A: {response_text}")
        
        # Add keywords if requested
        if include_keywords and keywords:
            parts.append(f"(Category: {keywords})")
        
        # Add question ID if verbose
        if verbose:
            parts.append(f"[{question_id}]")
        
        return " ".join(parts)
    
    def _map_response_to_text(self, question_info: Dict, response_value: any) -> str:
        """
        Map a response value to its text representation.
        
        Args:
            question_info: Question metadata dictionary
            response_value: The user's response
            
        Returns:
            Text representation of the response
        """
        # For multiple choice questions, map to option text
        try:
            response_int = int(float(response_value))
            
            # Check available options
            option_keys = [f'option_{i}' for i in range(1, 6)]  # Up to 5 options
            available_options = []
            
            for key in option_keys:
                option_text = question_info.get(key)
                if option_text and str(option_text).strip() and str(option_text).strip().lower() != 'nan':
                    available_options.append(option_text.strip())
            
            # Map 1-based index to option text
            if 1 <= response_int <= len(available_options):
                return available_options[response_int - 1]
            
        except (ValueError, TypeError):
            pass
        
        # For text responses or unmapped values, return as-is
        return str(response_value)
    
    def _format_demographics(self, user_row: pd.Series) -> str:
        """
        Format demographic information into natural language.
        
        Args:
            user_row: User data row
            
        Returns:
            Formatted demographic information
        """
        demo_parts = []
        
        # Key demographic mappings
        demo_mappings = {
            'd_age': ('Age', lambda x: f"{x} years old"),
            'd_gender': ('Gender', lambda x: str(x)),
            'd_orientation': ('Sexual Orientation', lambda x: str(x)),
            'd_ethnicity': ('Ethnicity', lambda x: str(x)),
            'd_education_type': ('Education', lambda x: str(x)),
            'd_job': ('Occupation', lambda x: str(x)),
            'd_income': ('Income Level', lambda x: str(x)),
            'd_relationship': ('Relationship Status', lambda x: str(x)),
            'd_religion_type': ('Religion', lambda x: str(x)),
            'd_bodytype': ('Body Type', lambda x: str(x)),
            'd_drinks': ('Drinking', lambda x: str(x)),
            'd_smokes': ('Smoking', lambda x: str(x)),
            'd_drugs': ('Drug Use', lambda x: str(x)),
            'd_offspring_current': ('Has Children', lambda x: str(x)),
            'd_offspring_desires': ('Wants Children', lambda x: str(x))
        }
        
        for column, (label, formatter) in demo_mappings.items():
            if column in user_row.index and pd.notna(user_row[column]):
                try:
                    formatted_value = formatter(user_row[column])
                    demo_parts.append(f"{label}: {formatted_value}")
                except:
                    # Skip if formatting fails
                    continue
        
        return "\n".join(demo_parts) if demo_parts else "No demographic information available"
    
    def format_batch(self, 
                    user_rows: pd.DataFrame, 
                    **format_kwargs) -> List[str]:
        """
        Format multiple user rows into text descriptions.
        
        Args:
            user_rows: DataFrame containing multiple user rows
            **format_kwargs: Arguments to pass to format_user_row
            
        Returns:
            List of formatted text descriptions
        """
        formatted_texts = []
        
        for idx, row in user_rows.iterrows():
            try:
                text = self.format_user_row(row, **format_kwargs)
                formatted_texts.append(text)
            except Exception as e:
                print(f"Warning: Failed to format row {idx}: {e}")
                formatted_texts.append(f"Error formatting user data: {str(e)}")
        
        return formatted_texts
    
    def get_sample_formatted_text(self, 
                                 n_samples: int = 3,
                                 **loader_kwargs) -> List[Tuple[str, float]]:
        """
        Get sample formatted texts with their true ages for testing.
        
        Args:
            n_samples: Number of samples to return
            **loader_kwargs: Arguments to pass to data_loader.get_rows()
            
        Returns:
            List of (formatted_text, true_age) tuples
        """
        # Get sample data
        features, ages = self.data_loader.get_rows(n_samples=n_samples, **loader_kwargs)
        
        # Format the texts
        formatted_texts = self.format_batch(features)
        
        return list(zip(formatted_texts, ages.values))


def main():
    """Example usage of the TextFormatter class."""
    # Initialize components
    data_loader = UserDataLoader()
    formatter = TextFormatter(data_loader)
    
    print("Getting sample formatted texts...")
    
    # Get a few sample formatted texts
    samples = formatter.get_sample_formatted_text(
        n_samples=2,
        min_age=18,
        max_age=65
    )
    
    for i, (text, age) in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1} (True Age: {age})")
        print('='*60)
        print(text)

if __name__ == "__main__":
    main()
