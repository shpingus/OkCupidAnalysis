#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage of LLM Age Guesser

This script demonstrates how to use the LLM Age Guesser package to predict
user ages from survey data using Google's Gemini API.

Before running:
1. Set your Google API key: export GOOGLE_API_KEY="your_api_key_here"
2. Ensure data files are in the correct location
3. Install required packages: pip install google-generativeai pandas python-dotenv
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_guesser import AgePredictionPipeline, UserDataLoader, TextFormatter, LLMPredictor
from dotenv import load_dotenv
load_dotenv()

def basic_usage_example():
    """
    Basic example: Predict ages for a small sample of users.
    """
    print("🚀 BASIC USAGE EXAMPLE")
    print("=" * 50)
    
    try:
        # Initialize the complete pipeline
        pipeline = AgePredictionPipeline(
            model_name="gemini-2.5-flash-lite"  # Fast and cost-effective model
        )
        
        # Run predictions on 30 users
        results = pipeline.run_complete_pipeline(
            n_samples=30,
            max_questions=15,  # Limit to 15 questions per user
            save_results=True  # Save results to files
        )
        
        print(f"\n✅ Successfully predicted ages for {len(results['predictions'])} users!")
        
        # Show some example predictions
        print("\n📋 SAMPLE PREDICTIONS:")
        for i, (pred, true_age) in enumerate(zip(results['predictions'][:3], results['true_ages'][:3])):
            error = abs(pred.predicted_age - true_age) if pred.success else "N/A"
            print(f"   User {i+1}: True={true_age:.0f}, Predicted={pred.predicted_age:.0f}, Error={error}")
            print(f"           Confidence={pred.confidence:.2f}, Explanation: {pred.explanation[:60]}...")
        
    except Exception as e:
        print(f"❌ Error in basic usage: {e}")
        print("Make sure your Google API key is set and data files are available.")


def component_usage_example():
    """
    Example showing how to use individual components.
    """
    print("\n\n🔧 COMPONENT USAGE EXAMPLE")
    print("=" * 50)
    
    try:
        # 1. Load and explore data
        print("\n1️⃣  Loading data...")
        loader = UserDataLoader()
        summary = loader.get_data_summary()
        print(f"   Dataset has {summary['total_rows']:,} users and {summary['question_columns']} questions")
        print(f"   Age range: {summary['age_stats']['min']:.0f}-{summary['age_stats']['max']:.0f} years")
        
        # Get a small sample
        features, ages = loader.get_rows(n_samples=3)
        print(f"   Selected {len(features)} users for processing")
        
        # 2. Format data into text
        print("\n2️⃣  Formatting data to text...")
        formatter = TextFormatter(loader)
        texts = formatter.format_batch(features, max_questions=10)
        
        print(f"   Generated {len(texts)} text descriptions")
        print(f"   Sample text preview:")
        print("   " + "-" * 40)
        print(f"   {texts[0][:200]}...")
        print("   " + "-" * 40)
        
        # 3. Predict with LLM
        print("\n3️⃣  Predicting ages...")
        predictor = LLMPredictor(model_name="gemini-1.5-flash")
        
        # Predict for just the first user
        prediction = predictor.predict_age(texts[0])
        true_age = ages.iloc[0]
        
        print(f"   True age: {true_age:.0f} years")
        print(f"   Predicted age: {prediction.predicted_age:.0f} years")
        print(f"   Error: {abs(prediction.predicted_age - true_age):.1f} years")
        print(f"   Confidence: {prediction.confidence:.2f}")
        print(f"   Explanation: {prediction.explanation}")
        
    except Exception as e:
        print(f"❌ Error in component usage: {e}")


def advanced_usage_example():
    """
    Advanced example with custom parameters and evaluation.
    """
    print("\n\n⚡ ADVANCED USAGE EXAMPLE")
    print("=" * 50)
    
    try:
        # Custom pipeline configuration
        pipeline = AgePredictionPipeline(
            model_name="gemini-1.5-flash",
        )
        
        print("\n🔧 Running with custom parameters...")
        results = pipeline.run_complete_pipeline(
            n_samples=10,
            columns=None,  # Let pipeline choose optimal columns  
            max_questions=25,  # More questions for better accuracy
            min_age=20,  # Focus on adults
            max_age=60,
            include_demographics=True,  # Include demographic info
            delay_seconds=0.2,  # Faster API calls
            save_results=True
        )
        
        eval_results = results['evaluation']
        
        print(f"\n📈 DETAILED EVALUATION:")
        print(f"   Model: gemini-1.5-flash")
        print(f"   Success rate: {eval_results['success_rate']*100:.1f}%")
        print(f"   Mean Absolute Error: {eval_results['mae']:.2f} years")
        print(f"   Accuracy within 10 years: {eval_results['accuracy_10y']*100:.1f}%")
        print(f"   Average confidence: {eval_results['mean_confidence']:.2f}")
        print(f"   Total processing time: {eval_results['total_time_seconds']:.1f} seconds")
        
        # Show confidence correlation with accuracy
        successful_preds = [p for p in results['predictions'] if p.success]
        if successful_preds:
            high_conf_preds = [p for p in successful_preds if p.confidence > 0.7]
            print(f"   High confidence predictions (>0.7): {len(high_conf_preds)}/{len(successful_preds)}")
        
    except Exception as e:
        print(f"❌ Error in advanced usage: {e}")


def main():
    """
    Run all usage examples.
    """
    print("🎯 LLM Age Guesser - Example Usage")
    print("=" * 60)
    print("This script demonstrates the LLM Age Guesser package capabilities.")
    print("Make sure you have set your GOOGLE_API_KEY environment variable!")
    print("=" * 60)
    
    # Check API key
    if not os.getenv('GOOGLE_API_KEY') and not os.getenv('GEMINI_API_KEY'):
        print("❌ ERROR: No Google API key found!")
        print("Please set your API key:")
        print("   export GOOGLE_API_KEY='your_api_key_here'")
        print("   OR")
        print("   export GEMINI_API_KEY='your_api_key_here'")
        return
    
    # Run examples
    try:
        basic_usage_example()
        component_usage_example() 
        advanced_usage_example()
        
        print(f"\n\n🎉 All examples completed successfully!")
        print("Check the 'results/' directory for saved prediction outputs.")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Examples interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
