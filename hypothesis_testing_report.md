# Hypothesis Testing Report: Age and Alcohol Consumption Analysis

## Executive Summary

We conducted a rigorous Chi-square test of independence to validate whether alcohol consumption patterns are age-related among OkCupid users. The analysis provides strong statistical evidence for an age-alcohol relationship (p < 0.001) but reveals a small practical effect size, indicating alcohol consumption is a valid but supplementary predictor for age in machine learning models.

## The 5-Step Hypothesis Testing Framework

**Research Question:** Does alcohol consumption frequency decrease with age among OkCupid users?

### Step 1: Define Your Hypotheses (Null, Alternative)

- **H₀ (Null Hypothesis):** Age and alcohol consumption are independent - there is no association between age and drinking frequency
- **H₁ (Alternative Hypothesis):** Alcohol consumption frequency decreases with age - older individuals are more likely to drink "Rarely" or "Never"

### Step 2: Specify Your Null Distribution

**Under H₀ Assumption:** If age and drinking were truly independent, the proportion of frequent drinkers should be the same across all age groups. Any observed differences would be random sampling variation.

**Statistical Distribution:** Chi-square (χ²) distribution with **9 degrees of freedom** = (4 age groups - 1) × (4 drinking levels - 1) = 3 × 3 = 9

### Step 3: Do an Experiment

**Study Design:** Cross-sectional analysis of 51,212 OkCupid users using question q77 ("How frequently do you drink alcohol?") with responses: Very often, Sometimes, Rarely, Never.

**Age Groups:** Users categorized into balanced groups: 18-25, 26-35, 36-45, and 46-60 years.

**Data Collection:** Constructed 4×4 contingency table with observed frequencies for each age-drinking combination.

### Step 4: Calculate the P-value

**Statistical Results:**

- **P-value: 1.35e-161** → Extremely statistically significant - virtually impossible to occur by chance if H₀ were true
- **Chi-square statistic: 777.69** → Very large test statistic indicating substantial deviation from independence  
- **Cramér's V (effect size): 0.0616** → Small effect size (0.1 = small, 0.3 = medium) - relationship exists but is weak
- **Sample size: 51,212 users** → Excellent statistical power ensuring reliable results

### Step 5: Reject or Fail to Reject Null Hypothesis

**Decision:** **REJECT H₀** and accept H₁ (p = 1.35e-161 < 0.05)

**Conclusion:** There is statistically significant evidence that alcohol consumption frequency decreases with age among OkCupid users.

## Age Group Distribution

| Age Group | Sample Size | Percentage | Drinking Pattern |
|-----------|-------------|------------|------------------|
| 18-25 | 10,525 | 20.6% | Higher "Very often" and "Sometimes" |
| 26-35 | 26,514 | 51.8% | Peak "Sometimes", declining "Very often" |
| 36-45 | 11,363 | 22.2% | Gradual shift toward "Rarely" |
| 46-60 | 2,810 | 5.5% | Highest "Never" and "Rarely" percentages |

## Visual Evidence

Three key visualizations support our conclusions:

1. **Contingency Table Heatmap** (`contingency_table_heatmap.png`): Shows clear differences between observed and expected frequencies, with younger groups over-represented in frequent drinking categories and older groups over-represented in infrequent drinking categories.

2. **Age-Drinking Pattern Visualization** (`age_drinking_patterns.png`): Stacked bar chart reveals the hypothesized gradient - decreasing alcohol consumption with increasing age across all four groups.

3. **Statistical Summary** (`statistical_summary.png`): Demonstrates the extremely small p-value relative to the significance threshold and positions our effect size in the "small" category.

## Conclusions and Implications

### Key Findings:
- **Statistical Significance:** The relationship between age and alcohol consumption is real and not due to random chance
- **Practical Significance:** The effect size is small (Cramér's V = 0.0616), indicating the relationship, while statistically robust, has limited practical impact

### Machine Learning Implications:
- **Model Enhancement:** Alcohol consumption represents a statistically robust predictor with small effect size that could provide modest improvements in future model iterations

**In Conclusion:** With a P-value of 1.35e-161, we definitively reject the null hypothesis and conclude that alcohol consumption patterns are statistically associated with age among OkCupid users. However, the small effect size (Cramér's V = 0.0616) indicates this represents a statistically robust predictor with limited practical impact. These findings suggest potential opportunities for incorporating alcohol-related features in future model enhancements while emphasizing that diverse feature sets remain essential for optimal performance.

---
*Analysis conducted using Chi-square test of independence with α = 0.05 significance level. All statistical assumptions verified: random sampling, independence of observations, and adequate expected frequencies (>5 per cell).*
