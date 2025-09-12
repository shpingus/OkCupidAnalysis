#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script to examine the OkCupid data structure
"""

import pandas as pd
import numpy as np

def examine_data_structure():
    """Examine the data structure to understand column names and formats"""
    
    print("Loading data to examine structure...")
    try:
        df = pd.read_csv('data/user_data_public.csv', low_memory=False)
        print(f"Dataset shape: {df.shape}")
        print(f"Column count: {len(df.columns)}")
        
        # Check if q77 column exists
        if 'q77' in df.columns:
            print("\n✓ q77 column found")
            print(f"q77 data type: {df['q77'].dtype}")
            print(f"q77 unique values: {sorted(df['q77'].dropna().unique())}")
            print(f"q77 value counts:")
            print(df['q77'].value_counts().head(10))
            print(f"q77 null count: {df['q77'].isnull().sum()}")
        else:
            print("\n❌ q77 column NOT found")
            # Look for alcohol-related columns
            alcohol_cols = [col for col in df.columns if 'alcohol' in col.lower() or 'drink' in col.lower()]
            print(f"Alcohol-related columns: {alcohol_cols}")
            
            # Look for q77 variations
            q77_cols = [col for col in df.columns if 'q77' in col.lower()]
            print(f"q77-related columns: {q77_cols}")
            
            # Show first few column names to understand structure
            print(f"First 20 columns: {list(df.columns[:20])}")
        
        # Check age column
        if 'd_age' in df.columns:
            print(f"\n✓ d_age column found")
            print(f"d_age data type: {df['d_age'].dtype}")
            print(f"d_age range: {df['d_age'].min()} to {df['d_age'].max()}")
            print(f"d_age null count: {df['d_age'].isnull().sum()}")
        else:
            print("\n❌ d_age column NOT found")
            age_cols = [col for col in df.columns if 'age' in col.lower()]
            print(f"Age-related columns: {age_cols}")
        
        # Look for question columns pattern
        q_cols = [col for col in df.columns if col.startswith('q') and col[1:].isdigit()]
        print(f"\nQuestion columns found: {len(q_cols)}")
        if len(q_cols) > 0:
            print(f"Sample question columns: {q_cols[:10]}")
            if 'q77' in q_cols:
                print("q77 is in question columns!")
            else:
                print("q77 NOT in question columns")
        
        # Check demographic columns
        demo_cols = [col for col in df.columns if col.startswith('d_')]
        print(f"\nDemographic columns found: {len(demo_cols)}")
        if len(demo_cols) > 0:
            print(f"Demographic columns: {demo_cols}")
            
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    examine_data_structure()
