#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Age Predictor using Google AI Studio API

This module handles the interaction with Google's Gemini API to predict user ages
based on their formatted question responses and demographic information.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re

try:
    import google.generativeai as genai
except ImportError as e:
    raise ImportError("Please install google-generativeai: pip install google-generativeai") from e

try:
    from tqdm import tqdm
except ImportError as e:
    raise ImportError("Please install tqdm: pip install tqdm") from e

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AgePrediction:
    """Structure for age prediction results."""
    predicted_age: float
    confidence: Optional[float]
    explanation: str
    raw_response: str
    success: bool
    error_message: Optional[str] = None


class LLMPredictor:
    """
    Handles age prediction using Google's Gemini AI models.
    
    This class manages API interactions, prompt formatting, response parsing,
    and error handling for LLM-based age prediction.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-2.5-flash",
                 temperature: float = 0.1,
                 max_output_tokens: int = 200):
        """
        Initialize the LLM predictor.
        
        Args:
            api_key: Google AI Studio API key. If None, loads from environment
            model_name: Gemini model to use
            temperature: Generation temperature (0.0-1.0)
            max_output_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Load API key
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY "
                "environment variable, or pass api_key parameter."
            )
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(model_name)
            print(f"Initialized {model_name} model successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model {model_name}: {e}")
    
    def create_age_prediction_prompt(self, user_text: str) -> str:
        """
        Create a structured prompt for age prediction.
        
        Args:
            user_text: Formatted user description
            
        Returns:
            Complete prompt for the LLM
        """
        prompt = f"""You are an expert demographic analyst tasked with predicting a person's age based on their survey responses and characteristics.

**Instructions:**
1. Analyze the provided information carefully
2. Consider patterns in responses, interests, values, and life circumstances
3. Provide a specific age estimate (as a number)
4. Give a brief explanation for your prediction
5. Be realistic - most online dating users are between 18-65 years old

**User Information:**
{user_text}

**Required Response Format (JSON):**
{{
    "predicted_age": [number between 18-100],
    "confidence": [number between 0.0-1.0],
    "explanation": "[brief explanation of reasoning]"
}}

**Important:** 
- Provide ONLY the JSON response, no additional text
- The predicted_age must be a specific number, not a range
- Confidence should reflect how certain you are (1.0 = very certain, 0.5 = moderate, 0.1 = uncertain)
- Keep explanation under 100 words

JSON Response:"""
        
        return prompt
    
    def predict_age(self, user_text: str, max_retries: int = 3) -> AgePrediction:
        """
        Predict age for a single user based on their text description.
        
        Args:
            user_text: Formatted user description
            max_retries: Maximum number of API call retries
            
        Returns:
            AgePrediction object with results
        """
        prompt = self.create_age_prediction_prompt(user_text)
        
        for attempt in range(max_retries):
            try:
                # Generate response
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                
                # Parse the response
                return self._parse_response(response.text, user_text[:200])
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return AgePrediction(
                        predicted_age=35.0,  # Default fallback
                        confidence=0.1,
                        explanation="Failed to get prediction from AI model",
                        raw_response="",
                        success=False,
                        error_message=str(e)
                    )
                
                # Wait before retry
                time.sleep(1)
        
        # Should not reach here
        return AgePrediction(
            predicted_age=35.0,
            confidence=0.0,
            explanation="Unknown error",
            raw_response="",
            success=False,
            error_message="Maximum retries exceeded"
        )
    
    def _parse_response(self, raw_response: str, user_preview: str) -> AgePrediction:
        """
        Parse the LLM response into structured format.
        
        Args:
            raw_response: Raw response text from the model
            user_preview: Preview of user data for error context
            
        Returns:
            AgePrediction object
        """
        try:
            # Clean up the response - look for JSON content
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = raw_response.strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Extract required fields with validation
            predicted_age = float(data.get('predicted_age', 35))
            confidence = float(data.get('confidence', 0.5))
            explanation = str(data.get('explanation', 'No explanation provided'))
            
            # Validate ranges
            predicted_age = max(18, min(100, predicted_age))  # Clamp to reasonable range
            confidence = max(0.0, min(1.0, confidence))      # Clamp to 0-1 range
            
            return AgePrediction(
                predicted_age=predicted_age,
                confidence=confidence,
                explanation=explanation,
                raw_response=raw_response,
                success=True
            )
            
        except json.JSONDecodeError as e:
            # Try to extract age from text if JSON parsing fails
            age_match = re.search(r'(?:age|Age)["\s]*:?\s*(\d+)', raw_response)
            predicted_age = float(age_match.group(1)) if age_match else 35.0
            
            return AgePrediction(
                predicted_age=predicted_age,
                confidence=0.3,
                explanation=f"Parsed from non-JSON response: {raw_response[:100]}...",
                raw_response=raw_response,
                success=False,
                error_message=f"JSON parsing failed: {e}"
            )
            
        except Exception as e:
            return AgePrediction(
                predicted_age=35.0,
                confidence=0.1,
                explanation="Failed to parse AI response",
                raw_response=raw_response,
                success=False,
                error_message=str(e)
            )
    
    def predict_batch(self, 
                     user_texts: List[str], 
                     delay_seconds: float = 0.5) -> List[AgePrediction]:
        """
        Predict ages for multiple users.
        
        Args:
            user_texts: List of formatted user descriptions
            delay_seconds: Delay between API calls to avoid rate limits
            
        Returns:
            List of AgePrediction objects
        """
        predictions = []
        
        # Use tqdm for progress bar
        with tqdm(total=len(user_texts), desc="Predicting ages", unit="user") as pbar:
            for i, user_text in enumerate(user_texts):
                prediction = self.predict_age(user_text)
                predictions.append(prediction)
                
                # Update progress bar with success info
                pbar.set_postfix({
                    'success_rate': f"{sum(p.success for p in predictions)}/{len(predictions)}",
                    'last_age': f"{prediction.predicted_age:.0f}" if prediction.success else "failed"
                })
                pbar.update(1)
                
                if i < len(user_texts) - 1:  # Don't delay after last prediction
                    time.sleep(delay_seconds)
        
        success_count = sum(p.success for p in predictions)
        print(f"Completed batch prediction. Success rate: {success_count}/{len(predictions)}")
        
        return predictions
    
    def evaluate_predictions(self, 
                           predictions: List[AgePrediction], 
                           true_ages: List[float]) -> Dict[str, float]:
        """
        Evaluate prediction accuracy against true ages.
        
        Args:
            predictions: List of age predictions
            true_ages: List of true ages
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(predictions) != len(true_ages):
            raise ValueError("Number of predictions and true ages must match")
        
        # Extract successful predictions
        valid_pairs = [
            (pred.predicted_age, true_age) 
            for pred, true_age in zip(predictions, true_ages) 
            if pred.success
        ]
        
        if not valid_pairs:
            return {
                'mae': float('inf'),
                'rmse': float('inf'),
                'mean_error': float('inf'),
                'accuracy_5y': 0.0,
                'accuracy_10y': 0.0,
                'success_rate': 0.0,
                'n_valid': 0
            }
        
        predicted_ages = [p[0] for p in valid_pairs]
        actual_ages = [p[1] for p in valid_pairs]
        errors = [abs(p - a) for p, a in valid_pairs]
        
        # Calculate metrics
        mae = sum(errors) / len(errors)
        rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
        mean_error = sum(p - a for p, a in valid_pairs) / len(valid_pairs)
        
        accuracy_5y = sum(1 for e in errors if e <= 5) / len(errors)
        accuracy_10y = sum(1 for e in errors if e <= 10) / len(errors)
        success_rate = len(valid_pairs) / len(predictions)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mean_error': mean_error,
            'accuracy_5y': accuracy_5y,
            'accuracy_10y': accuracy_10y,
            'success_rate': success_rate,
            'n_valid': len(valid_pairs)
        }


def main():
    """Example usage of the LLMPredictor class."""
    try:
        # Initialize predictor
        predictor = LLMPredictor()
        
        # Test with sample user description
        sample_text = """**Question Responses:**
Q: Do you enjoy meaningless sex?
A: No (Category: sex/intimacy)
Q: Would you date someone just for the sex?
A: No (Category: sex/intimacy)
Q: How does the idea of being slapped hard in the face during sex make you feel?
A: Horrified (Category: sex/intimacy)

**Demographics:**
Age: 28 years old
Gender: Man
Sexual Orientation: Straight
Education: university
Occupation: Teacher
Relationship Status: Single"""
        
        print("Testing age prediction with sample user...")
        print("Sample user preview:", sample_text[:200] + "...")
        
        # Make prediction
        result = predictor.predict_age(sample_text)
        
        print(f"\nPrediction Results:")
        print(f"Predicted Age: {result.predicted_age}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Success: {result.success}")
        print(f"Explanation: {result.explanation}")
        
        if not result.success:
            print(f"Error: {result.error_message}")
        
    except Exception as e:
        print(f"Error testing predictor: {e}")
        print("Make sure you have set your GOOGLE_API_KEY or GEMINI_API_KEY environment variable")

if __name__ == "__main__":
    main()
