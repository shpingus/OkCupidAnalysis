#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistical Hypothesis Testing: Age and Alcohol Consumption

This script implements a Chi-square test of independence to test whether
alcohol consumption frequency decreases with age among OkCupid users.

Research Question: Does alcohol consumption frequency decrease with age?
H₀: Age and alcohol consumption are independent
H₁: Alcohol consumption frequency decreases with age

Database Question: q77 - "How frequently do you drink alcohol?"
Response Options: 1=Very often, 2=Sometimes, 3=Rarely, 4=Never
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os
from pathlib import Path

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data(data_path='data/user_data_public.csv'):
    """
    Load OkCupid data and prepare it for hypothesis testing.
    
    Args:
        data_path: Path to the user data CSV file
        
    Returns:
        DataFrame with cleaned age and alcohol data
    """
    print(f"Loading data from {data_path}")
    
    try:
        # Load only the required columns to optimize memory usage
        print("Loading only required columns (d_age, q77) to optimize memory...")
        df = pd.read_csv(data_path, usecols=['d_age', 'q77'], low_memory=False)
        print(f"Loaded dataset shape: {df.shape}")
        
        # Show data types and sample values
        print(f"d_age data type: {df['d_age'].dtype}")
        print(f"q77 data type: {df['q77'].dtype}")
        print(f"q77 unique values: {sorted(df['q77'].dropna().unique())}")
        
        # Filter for users with valid age and q77 responses
        valid_data = df.dropna(subset=['d_age', 'q77'])
        print(f"Users with valid age and alcohol data: {len(valid_data)}")
        
        # Filter reasonable age range (18-80)
        valid_data = valid_data[(valid_data['d_age'] >= 18) & (valid_data['d_age'] <= 80)]
        print(f"Users in age range 18-80: {len(valid_data)}")
        
        # Filter valid q77 responses (string values)
        valid_alcohol_responses = ['Very often', 'Sometimes', 'Rarely', 'Never']
        valid_data = valid_data[valid_data['q77'].isin(valid_alcohol_responses)]
        print(f"Users with valid alcohol responses: {len(valid_data)}")
        
        return valid_data[['d_age', 'q77']].copy()
        
    except FileNotFoundError:
        print(f"Error: Could not find data file at {data_path}")
        print("Please ensure the OkCupid dataset is available in the data/ directory")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_age_groups(df):
    """
    Create age group categories for analysis.
    
    Args:
        df: DataFrame with age data
        
    Returns:
        DataFrame with age_group column added
    """
    def categorize_age(age):
        if 18 <= age <= 25:
            return '18-25'
        elif 26 <= age <= 35:
            return '26-35'
        elif 36 <= age <= 50:
            return '36-50'
        elif age > 50:
            return '50+'
        else:
            return 'Other'
    
    df['age_group'] = df['d_age'].apply(categorize_age)
    
    # Remove any 'Other' categories
    df = df[df['age_group'] != 'Other']
    
    print("\nAge group distribution:")
    print(df['age_group'].value_counts().sort_index())
    
    return df

def map_alcohol_responses(df):
    """
    Process alcohol responses (already in string format) and create alcohol_frequency column.
    
    Args:
        df: DataFrame with q77 responses (string values)
        
    Returns:
        DataFrame with alcohol_frequency column added
    """
    # Data is already in string format, just copy to alcohol_frequency column
    df['alcohol_frequency'] = df['q77'].copy()
    
    print("\nAlcohol frequency distribution:")
    print(df['alcohol_frequency'].value_counts())
    
    return df

def create_contingency_table(df):
    """
    Create contingency table for age groups vs alcohol frequency.
    
    Args:
        df: DataFrame with age_group and alcohol_frequency columns
        
    Returns:
        Contingency table as DataFrame
    """
    # Create contingency table
    contingency_table = pd.crosstab(
        df['age_group'], 
        df['alcohol_frequency'],
        margins=True,
        margins_name='Total'
    )
    
    # Reorder columns for logical progression
    column_order = ['Very often', 'Sometimes', 'Rarely', 'Never', 'Total']
    contingency_table = contingency_table[column_order]
    
    # Reorder rows for age progression
    row_order = ['18-25', '26-35', '36-50', '50+', 'Total']
    contingency_table = contingency_table.reindex(row_order)
    
    print("\nContingency Table (Observed Frequencies):")
    print("=" * 60)
    print(contingency_table)
    
    return contingency_table

def calculate_percentages(contingency_table):
    """
    Calculate row percentages for the contingency table.
    
    Args:
        contingency_table: Contingency table DataFrame
        
    Returns:
        DataFrame with row percentages
    """
    # Calculate row percentages (excluding totals)
    data_only = contingency_table.iloc[:-1, :-1]  # Exclude Total row and column
    row_percentages = data_only.div(data_only.sum(axis=1), axis=0) * 100
    
    print("\nRow Percentages (% within each age group):")
    print("=" * 60)
    print(row_percentages.round(1))
    
    return row_percentages

def perform_chi_square_test(contingency_table):
    """
    Perform Chi-square test of independence.
    
    Args:
        contingency_table: Contingency table DataFrame
        
    Returns:
        Dictionary with test results
    """
    # Extract data without totals for the test
    observed = contingency_table.iloc[:-1, :-1].values
    
    # Perform Chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(observed)
    
    # Calculate Cramér's V (effect size)
    n = observed.sum()
    cramers_v = np.sqrt(chi2_stat / (n * min(observed.shape) - 1))
    
    # Create results dictionary
    results = {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected,
        'cramers_v': cramers_v,
        'sample_size': n
    }
    
    print("\nChi-Square Test Results:")
    print("=" * 60)
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Degrees of freedom: {dof}")
    print(f"Sample size: {n:,}")
    print(f"Cramér's V (effect size): {cramers_v:.4f}")
    
    return results

def interpret_results(results, alpha=0.05):
    """
    Interpret the statistical test results.
    
    Args:
        results: Dictionary with test results
        alpha: Significance level (default 0.05)
    """
    print("\nStatistical Interpretation:")
    print("=" * 60)
    
    # Hypothesis test decision
    if results['p_value'] < alpha:
        print(f"✓ REJECT H₀ (p = {results['p_value']:.2e} < {alpha})")
        print("✓ ACCEPT H₁: Alcohol consumption frequency decreases with age")
        print("✓ Result is STATISTICALLY SIGNIFICANT")
    else:
        print(f"✗ FAIL TO REJECT H₀ (p = {results['p_value']:.4f} ≥ {alpha})")
        print("✗ No strong evidence that alcohol consumption decreases with age")
        print("✗ Result is NOT statistically significant")
    
    # Effect size interpretation
    cramers_v = results['cramers_v']
    if cramers_v < 0.1:
        effect_size = "negligible"
    elif cramers_v < 0.3:
        effect_size = "small"
    elif cramers_v < 0.5:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    print(f"\nEffect Size: {effect_size} (Cramér's V = {cramers_v:.4f})")
    
    # Age prediction implications
    print("\nAge Prediction Model Implications:")
    print("-" * 40)
    if results['p_value'] < alpha and cramers_v > 0.1:
        print("✓ Alcohol consumption is a VALID age predictor")
        print("✓ Supports using q77 as a feature in ML models")
        print("✓ Age-alcohol relationship is statistically robust")
    else:
        print("? Weak evidence for alcohol as an age predictor")
        print("? Consider reducing weight of alcohol features in ML models")

def create_visualizations(contingency_table, row_percentages, results):
    """
    Create visualizations for the hypothesis test results.
    
    Args:
        contingency_table: Contingency table DataFrame
        row_percentages: Row percentages DataFrame
        results: Test results dictionary
    """
    # Create results directory
    results_dir = Path('hypothesis_results')
    results_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Contingency Table Heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Observed frequencies heatmap
    observed_data = contingency_table.iloc[:-1, :-1]  # Exclude totals
    sns.heatmap(observed_data, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Observed Frequencies\n(Actual Data)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Alcohol Consumption Frequency', fontsize=12)
    ax1.set_ylabel('Age Group', fontsize=12)
    
    # Expected frequencies heatmap
    expected_df = pd.DataFrame(
        results['expected_frequencies'],
        index=observed_data.index,
        columns=observed_data.columns
    )
    sns.heatmap(expected_df, annot=True, fmt='.1f', cmap='Oranges', ax=ax2)
    ax2.set_title('Expected Frequencies\n(Under H₀: Independence)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Alcohol Consumption Frequency', fontsize=12)
    ax2.set_ylabel('Age Group', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'contingency_table_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {results_dir}/contingency_table_heatmap.png")
    plt.close()
    
    # 2. Age-Drinking Pattern Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create stacked bar chart
    row_percentages.plot(kind='bar', stacked=True, ax=ax, 
                        color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71'])
    
    ax.set_title('Alcohol Consumption Patterns by Age Group\n' + 
                f'Chi-square test: p = {results["p_value"]:.2e}, ' +
                f'Cramér\'s V = {results["cramers_v"]:.3f}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Percentage within Age Group', fontsize=12)
    ax.legend(title='Alcohol Frequency', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Add horizontal line at 25% for reference
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5, 
               label='25% (Equal distribution)')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'age_drinking_patterns.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {results_dir}/age_drinking_patterns.png")
    plt.close()
    
    # 3. Statistical Summary Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Age distribution
    age_counts = contingency_table.iloc[:-1, -1]  # Total column, exclude grand total
    ax1.bar(age_counts.index, age_counts.values, color='skyblue', alpha=0.7)
    ax1.set_title('Sample Size by Age Group', fontweight='bold')
    ax1.set_ylabel('Number of Users')
    for i, v in enumerate(age_counts.values):
        ax1.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Alcohol distribution
    alcohol_counts = contingency_table.iloc[-1, :-1]  # Total row, exclude grand total
    ax2.bar(alcohol_counts.index, alcohol_counts.values, color='lightcoral', alpha=0.7)
    ax2.set_title('Sample Size by Alcohol Frequency', fontweight='bold')
    ax2.set_ylabel('Number of Users')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(alcohol_counts.values):
        ax2.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # P-value visualization
    p_val = results['p_value']
    colors = ['red' if p_val < 0.05 else 'gray']
    ax3.bar(['P-value'], [p_val], color=colors, alpha=0.7)
    ax3.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    ax3.set_title('P-value vs Significance Threshold', fontweight='bold')
    ax3.set_ylabel('P-value')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.text(0, p_val * 2, f'{p_val:.2e}', ha='center', fontweight='bold')
    
    # Effect size visualization
    cramers_v = results['cramers_v']
    effect_thresholds = [0.1, 0.3, 0.5]
    effect_labels = ['Small', 'Medium', 'Large']
    colors = ['lightgreen', 'yellow', 'orange', 'red']
    
    bars = ax4.bar(range(len(effect_thresholds) + 1), 
                   effect_thresholds + [1.0], 
                   color=colors, alpha=0.3, width=0.8)
    ax4.bar([0], [cramers_v], color='darkblue', alpha=0.8, width=0.4, 
            label=f'Our Result: {cramers_v:.3f}')
    ax4.set_title('Effect Size (Cramér\'s V)', fontweight='bold')
    ax4.set_ylabel('Cramér\'s V')
    ax4.set_xticks(range(len(effect_labels) + 1))
    ax4.set_xticklabels(['Negligible'] + effect_labels)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / 'statistical_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {results_dir}/statistical_summary.png")
    plt.close()

def save_detailed_results(contingency_table, results, row_percentages):
    """
    Save detailed results to text files.
    
    Args:
        contingency_table: Contingency table DataFrame
        results: Test results dictionary
        row_percentages: Row percentages DataFrame
    """
    results_dir = Path('hypothesis_results')
    results_dir.mkdir(exist_ok=True)
    
    # Save statistical results
    with open(results_dir / 'hypothesis_test_results.txt', 'w') as f:
        f.write("STATISTICAL HYPOTHESIS TESTING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Research Question: Does alcohol consumption frequency decrease with age?\n")
        f.write("Database Question: q77 - 'How frequently do you drink alcohol?'\n")
        f.write("Sample Size: {:,} OkCupid users\n\n".format(results['sample_size']))
        
        f.write("HYPOTHESES:\n")
        f.write("H₀: Age and alcohol consumption are independent\n")
        f.write("H₁: Alcohol consumption frequency decreases with age\n\n")
        
        f.write("CHI-SQUARE TEST RESULTS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Chi-square statistic: {results['chi2_statistic']:.4f}\n")
        f.write(f"P-value: {results['p_value']:.2e}\n")
        f.write(f"Degrees of freedom: {results['degrees_of_freedom']}\n")
        f.write(f"Cramér's V (effect size): {results['cramers_v']:.4f}\n\n")
        
        f.write("STATISTICAL DECISION:\n")
        f.write("-" * 20 + "\n")
        if results['p_value'] < 0.05:
            f.write("REJECT H₀ - Accept H₁\n")
            f.write("Conclusion: Statistically significant evidence that alcohol consumption decreases with age\n")
        else:
            f.write("FAIL TO REJECT H₀\n")
            f.write("Conclusion: No statistically significant evidence of age-alcohol relationship\n")
        
        f.write(f"\nEffect Size Interpretation: ")
        if results['cramers_v'] < 0.1:
            f.write("Negligible effect\n")
        elif results['cramers_v'] < 0.3:
            f.write("Small effect\n")
        elif results['cramers_v'] < 0.5:
            f.write("Medium effect\n")
        else:
            f.write("Large effect\n")
        
        f.write("\nAGE PREDICTION MODEL IMPLICATIONS:\n")
        f.write("-" * 35 + "\n")
        if results['p_value'] < 0.05 and results['cramers_v'] > 0.1:
            f.write("✓ Alcohol consumption is a valid age predictor\n")
            f.write("✓ Supports using q77 as a feature in ML models\n")
        else:
            f.write("? Weak evidence for alcohol as an age predictor\n")
    
    print(f"Saved: {results_dir}/hypothesis_test_results.txt")
    
    # Save contingency table
    contingency_table.to_csv(results_dir / 'contingency_table.csv')
    print(f"Saved: {results_dir}/contingency_table.csv")
    
    # Save row percentages
    row_percentages.to_csv(results_dir / 'row_percentages.csv')
    print(f"Saved: {results_dir}/row_percentages.csv")

def main():
    """
    Main execution function for the hypothesis test.
    """
    print("STATISTICAL HYPOTHESIS TESTING: AGE AND ALCOHOL CONSUMPTION")
    print("=" * 65)
    print("Research Question: Does alcohol consumption frequency decrease with age?")
    print("Database Question: q77 - 'How frequently do you drink alcohol?'")
    print()
    
    # Step 1: Load and prepare data
    print("STEP 1: Loading and preparing data...")
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Step 2: Create age groups and map alcohol responses
    print("\nSTEP 2: Creating age groups and mapping responses...")
    df = create_age_groups(df)
    df = map_alcohol_responses(df)
    
    # Step 3: Create contingency table
    print("\nSTEP 3: Creating contingency table...")
    contingency_table = create_contingency_table(df)
    row_percentages = calculate_percentages(contingency_table)
    
    # Step 4: Perform statistical test
    print("\nSTEP 4: Performing Chi-square test...")
    results = perform_chi_square_test(contingency_table)
    
    # Step 5: Interpret results
    print("\nSTEP 5: Interpreting results...")
    interpret_results(results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(contingency_table, row_percentages, results)
    
    # Save detailed results
    print("\nSaving detailed results...")
    save_detailed_results(contingency_table, results, row_percentages)
    
    print("\n" + "=" * 65)
    print("HYPOTHESIS TESTING COMPLETE!")
    print("Check the 'hypothesis_results/' directory for detailed outputs.")
    print("=" * 65)

if __name__ == "__main__":
    main()
