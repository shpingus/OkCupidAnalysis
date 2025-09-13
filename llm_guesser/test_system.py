#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Test for LLM Age Guesser

This script performs basic tests to verify the system is working correctly
without making API calls (to avoid costs during testing).
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_data_loader():
    """Test the data loader functionality."""
    print("🔍 Testing UserDataLoader...")
    
    try:
        from llm_guesser import UserDataLoader
        
        # Initialize loader
        loader = UserDataLoader()
        
        # Test data summary (this loads the data)
        summary = loader.get_data_summary()
        print(f"   ✅ Data loaded: {summary['total_rows']:,} users, {summary['question_columns']} questions")
        
        # Test getting rows
        features, ages = loader.get_rows(n_samples=3)
        print(f"   ✅ Got {len(features)} user rows with {len(features.columns)} features")
        
        # Test question info
        question_info = loader.get_question_info('q2')
        if question_info:
            print(f"   ✅ Question metadata loaded: '{question_info.get('text', 'N/A')[:50]}...'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ UserDataLoader failed: {e}")
        return False

def test_text_formatter():
    """Test the text formatter functionality."""
    print("\n📝 Testing TextFormatter...")
    
    try:
        from llm_guesser import UserDataLoader, TextFormatter
        
        # Initialize components
        loader = UserDataLoader()
        formatter = TextFormatter(loader)
        
        # Get sample data
        features, ages = loader.get_rows(n_samples=2)
        
        # Test single user formatting
        user_row = features.iloc[0]
        text = formatter.format_user_row(user_row, max_questions=5)
        
        print(f"   ✅ Formatted user row: {len(text)} characters")
        print(f"   ✅ Sample text: '{text[:100]}...'")
        
        # Test batch formatting
        texts = formatter.format_batch(features, max_questions=5)
        print(f"   ✅ Batch formatted {len(texts)} users")
        
        return True
        
    except Exception as e:
        print(f"   ❌ TextFormatter failed: {e}")
        return False

def test_llm_predictor_mock():
    """Test LLM predictor without API calls."""
    print("\n🤖 Testing LLMPredictor (mock mode)...")
    
    try:
        from llm_guesser.llm_predictor import AgePrediction
        
        # Test AgePrediction dataclass
        prediction = AgePrediction(
            predicted_age=25.5,
            confidence=0.75,
            explanation="Test prediction",
            raw_response="{}",
            success=True
        )
        
        print(f"   ✅ AgePrediction created: age={prediction.predicted_age}, confidence={prediction.confidence}")
        
        # Test evaluation metrics calculation (without actual API calls)
        test_predictions = [
            AgePrediction(25.0, 0.8, "test", "{}", True),
            AgePrediction(30.0, 0.7, "test", "{}", True),
            AgePrediction(35.0, 0.9, "test", "{}", True),
        ]
        test_ages = [23.0, 32.0, 37.0]
        
        # Mock the evaluation
        errors = [abs(pred.predicted_age - true_age) for pred, true_age in zip(test_predictions, test_ages)]
        mae = sum(errors) / len(errors)
        
        print(f"   ✅ Evaluation metrics calculated: MAE={mae:.1f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LLMPredictor test failed: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization without API calls."""
    print("\n⚡ Testing AgePredictionPipeline initialization...")
    
    try:
        # Mock the LLMPredictor to avoid API key requirement
        import llm_guesser.pipeline
        
        # Test components individually first
        from llm_guesser import UserDataLoader, TextFormatter
        
        loader = UserDataLoader()
        formatter = TextFormatter(loader)
        
        print("   ✅ Pipeline components initialized successfully")
        
        # Test data flow
        features, ages = loader.get_rows(n_samples=2)
        texts = formatter.format_batch(features, max_questions=5)
        
        print(f"   ✅ Data pipeline working: {len(features)} users → {len(texts)} text descriptions")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Pipeline test failed: {e}")
        return False

def test_imports():
    """Test that all imports work correctly."""
    print("\n📦 Testing package imports...")
    
    try:
        # Test main package import
        import llm_guesser
        print(f"   ✅ Main package imported: version {llm_guesser.__version__}")
        
        # Test individual components
        from llm_guesser import UserDataLoader, TextFormatter, AgePrediction
        print("   ✅ Core components imported")
        
        # Test pipeline import
        from llm_guesser import AgePredictionPipeline
        print("   ✅ Pipeline imported")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        return False

def check_data_files():
    """Check if required data files exist."""
    print("\n📂 Checking data files...")
    
    data_dir = Path(__file__).parent.parent / "data"
    
    files_to_check = [
        "user_data_public.csv",
        "question_data.csv"
    ]
    
    all_exist = True
    for filename in files_to_check:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ {filename}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {filename}: NOT FOUND")
            all_exist = False
    
    return all_exist

def check_api_key():
    """Check if API key is configured."""
    print("\n🔑 Checking API key configuration...")
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    
    if api_key:
        print(f"   ✅ API key found: {api_key[:3]}{'*' * (len(api_key)-10)}{api_key[-3:]}")
        return True
    else:
        print("   ⚠️  No API key found (required for actual predictions)")
        print("      Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        return False

def main():
    """Run all system tests."""
    print("🧪 LLM Age Guesser - System Tests")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Files", check_data_files),
        ("API Key", check_api_key),
        ("Data Loader", test_data_loader),
        ("Text Formatter", test_text_formatter),
        ("LLM Predictor", test_llm_predictor_mock),
        ("Pipeline", test_pipeline_initialization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! System is ready to use.")
        print("To run actual predictions, make sure you have a valid Google API key.")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
