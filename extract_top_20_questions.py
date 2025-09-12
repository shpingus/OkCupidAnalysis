#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract Top 20 Questions for Statistical Hypothesis Testing

This script extracts the top 20 questions by response count from the OkCupid 
question dataset and writes them to a formatted file for easy review.
The goal is to identify the most promising questions for age-related 
statistical hypothesis testing.
"""

import pandas as pd
import os

def extract_top_20_questions(question_data_path='data/question_data.csv', 
                           output_file='top_20_questions_for_hypothesis_testing.txt'):
    """
    Extract the top 20 questions by response count and write to a formatted file.
    
    Args:
        question_data_path: Path to the question data CSV file
        output_file: Path for the output file
    """
    
    print(f"Loading question data from {question_data_path}")
    
    # Load the question data
    try:
        question_csv = pd.read_csv(question_data_path, delimiter=';', low_memory=False, index_col=0)
        print(f"Successfully loaded {len(question_csv)} questions")
    except Exception as e:
        print(f"Error loading question data: {e}")
        return
    
    # Get high-response questions (N > 50,000) - same logic as in the model
    high_response_questions = question_csv[question_csv['N'] > 50000]
    print(f"Found {len(high_response_questions)} questions with > 50,000 responses")
    
    # Get the top 20 questions by response count
    # Since the data is already sorted by response count, we take the first 20
    top_20_questions = high_response_questions.head(20)
    
    print(f"Extracting top {len(top_20_questions)} questions for hypothesis testing")
    
    # Write to formatted file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TOP 20 OKCUPID QUESTIONS FOR AGE-RELATED HYPOTHESIS TESTING\n")
        f.write("=" * 70 + "\n\n")
        f.write("These are the 20 questions with the highest response counts (N > 50,000)\n")
        f.write("from the OkCupid dataset, selected for statistical hypothesis testing\n")
        f.write("to explore age-related differences in responses.\n\n")
        f.write(f"Total questions analyzed: {len(question_csv)}\n")
        f.write(f"Questions with >50k responses: {len(high_response_questions)}\n")
        f.write(f"Selected for analysis: {len(top_20_questions)}\n\n")
        f.write("=" * 70 + "\n\n")
        
        # Write each question with details
        for idx, (question_id, row) in enumerate(top_20_questions.iterrows(), 1):
            f.write(f"QUESTION {idx}: {question_id}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Response Count (N): {row['N']:,}\n")
            
            # Question text
            if pd.notna(row.get('text', '')):
                f.write(f"Question Text: {row['text']}\n")
            else:
                f.write("Question Text: [Not available]\n")
            
            # Category (check what columns are actually available)
            category_cols = [col for col in row.index if 'cat' in col.lower() or 'type' in col.lower()]
            if category_cols:
                category_value = row[category_cols[0]]
                if pd.notna(category_value):
                    f.write(f"Category: {category_value}\n")
                else:
                    f.write("Category: [Not specified]\n")
            else:
                f.write("Category: [Not available]\n")
            
            # Response options (if available)
            option_columns = [col for col in row.index if col.startswith('option')]
            if option_columns:
                f.write("Response Options:\n")
                for opt_col in option_columns:
                    if pd.notna(row[opt_col]) and str(row[opt_col]).strip():
                        option_num = opt_col.replace('option', '')
                        f.write(f"  {option_num}: {row[opt_col]}\n")
            else:
                f.write("Response Options: [Not available]\n")
            
            # Potential hypothesis ideas (placeholder for user to fill)
            f.write("\nPotential Age-Related Hypotheses:\n")
            f.write("  [ ] Hypothesis idea 1: _________________________________\n")
            f.write("  [ ] Hypothesis idea 2: _________________________________\n")
            f.write("  [ ] Other notes: ____________________________________\n")
            
            f.write("\n" + "=" * 70 + "\n\n")
        
        # Add summary section for hypothesis planning
        f.write("HYPOTHESIS TESTING PLANNING NOTES\n")
        f.write("=" * 70 + "\n\n")
        f.write("Selected Questions for Testing (mark with X):\n")
        for idx, (question_id, row) in enumerate(top_20_questions.iterrows(), 1):
            f.write(f"  [ ] Q{idx} ({question_id}): {row.get('text', 'No text')[:50]}...\n")
        
        f.write("\nStatistical Test Framework:\n")
        f.write("1. Define hypotheses (H0 and H1)\n")
        f.write("2. Specify null distribution\n") 
        f.write("3. Design experiment/analysis\n")
        f.write("4. Calculate p-value\n")
        f.write("5. Reject or fail to reject null hypothesis\n\n")
        
        f.write("Age Group Considerations:\n")
        f.write("  - Young adults (18-25)\n")
        f.write("  - Adults (26-35)\n")
        f.write("  - Middle-aged (36-50)\n")
        f.write("  - Older adults (50+)\n\n")
        
        f.write("Potential Statistical Tests:\n")
        f.write("  - Chi-square test (categorical responses)\n")
        f.write("  - ANOVA (comparing means across age groups)\n")
        f.write("  - Correlation analysis (age as continuous variable)\n")
        f.write("  - Logistic regression (binary outcomes)\n\n")
    
    print(f"\nTop 20 questions successfully written to: {output_file}")
    print(f"File contains {len(top_20_questions)} questions with detailed information")
    print("You can now review this file to select the most interesting questions for hypothesis testing!")
    
    # Print a quick summary to console
    print(f"\nQuick Summary of Top 5 Questions:")
    print("-" * 50)
    for idx, (question_id, row) in enumerate(top_20_questions.head(5).iterrows(), 1):
        response_count = f"{row['N']:,}"
        question_text = row.get('text', 'No text available')[:60]
        print(f"{idx}. {question_id} (N={response_count}): {question_text}...")
    
    return top_20_questions

if __name__ == "__main__":
    # Extract the top 20 questions
    top_questions = extract_top_20_questions()
    
    if top_questions is not None:
        print(f"\nExtraction completed successfully!")
        print(f"Review the file 'top_20_questions_for_hypothesis_testing.txt' to select")
        print(f"the most interesting questions for your age-related statistical analysis.")
