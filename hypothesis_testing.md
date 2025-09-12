# Statistical Hypothesis Testing: Age and Alcohol Consumption

## Project Context
This analysis is part of the OkCupid Age Prediction project, where we use machine learning models to predict user age based on demographic and question response data. This hypothesis test validates whether alcohol consumption patterns are indeed age-related, supporting the use of alcohol-related features in our age prediction models.

## Research Question
**Does alcohol consumption frequency decrease with age among OkCupid users?**

## Database Question Used
**Question ID:** q77  
**Question Text:** "How frequently do you drink alcohol?"  
**Response Options:**
- 1 = Very often
- 2 = Sometimes  
- 3 = Rarely
- 4 = Never

**Sample Size:** 52,467 responses (one of the highest response rates in the dataset)

---

## The 5-Step Hypothesis Testing Framework

### Step 1: Define Your Hypotheses (Null, Alternative)

**H₀ (Null Hypothesis):** Age and alcohol consumption frequency are independent. There is no association between a person's age and how frequently they drink alcohol.

**H₁ (Alternative Hypothesis):** Alcohol consumption frequency decreases with age. Older individuals are more likely to drink "Rarely" or "Never" compared to younger individuals.

**Mathematical Formulation:**
- H₀: P(Drinking Level | Age Group) = P(Drinking Level) for all age groups
- H₁: P("Very often" | Young) > P("Very often" | Old) AND P("Never" | Old) > P("Never" | Young)

**Age Groups Defined:**
- Young Adults: 18-25 years
- Adults: 26-35 years  
- Middle-aged: 36-50 years
- Older Adults: 50+ years

### Step 2: Specify Your Null Distribution

**Under H₀ Assumption:** If age and drinking are truly independent, then:
- The proportion of "Very often" drinkers should be the same across all age groups
- The proportion of "Never" drinkers should be the same across all age groups
- Any observed differences are just random sampling variation

**Statistical Distribution:** Chi-square (χ²) distribution with degrees of freedom = (number of age groups - 1) × (number of drinking levels - 1) = (4-1) × (4-1) = 9

**Expected Frequencies Calculation:**
For each cell in the contingency table:
Expected Frequency = (Row Total × Column Total) / Grand Total

**Example:** If 25% of all users are aged 18-25 and 15% of all users drink "Very often", then under H₀, we expect 25% × 15% = 3.75% of all users to be young frequent drinkers.

### Step 3: Do an Experiment

**Our Observational Study Design:**
1. Load OkCupid user dataset with age and q77 response data
2. Filter users with valid ages (18-80) and valid alcohol responses
3. Create age group categories: 18-25, 26-35, 36-50, 50+
4. Map q77 numeric responses to descriptive labels
5. Construct 4×4 contingency table with observed frequencies
6. Calculate row totals, column totals, and grand total

**Expected Pattern if H₁ is True:**
- **Young Adults (18-25):** Higher percentages of "Very often" and "Sometimes"
- **Adults (26-35):** Peak "Sometimes", declining "Very often"
- **Middle-aged (36-50):** Gradual shift toward "Rarely"
- **Older Adults (50+):** Highest percentages of "Never" and "Rarely"

### Step 4: Calculate the P-value

**Test Statistic:** Chi-square statistic
χ² = Σ[(Observed - Expected)² / Expected] for all 16 cells

**P-value Calculation:** 
The probability of observing our χ² statistic (or a more extreme value) if H₀ is true, using the χ² distribution with 9 degrees of freedom.

**P-value Interpretation:**
- **Small p-value (< 0.05):** Our observed age-drinking pattern is very unlikely under H₀
- **Large p-value (≥ 0.05):** Our observed pattern is plausible under H₀

**What P-value Means in Context:**
- p = 0.001: "If age truly doesn't affect drinking, there's only a 0.1% chance we'd see this strong of an age-drinking pattern by random chance"
- p = 0.15: "If age truly doesn't affect drinking, there's a 15% chance we'd see this pattern by random chance"

### Step 5: Reject or Fail to Reject Null Hypothesis

**Decision Rule:** α = 0.05 significance level (95% confidence threshold)

**If p-value < 0.05:**
- **Decision:** Reject H₀, accept H₁
- **Conclusion:** "There is statistically significant evidence that alcohol consumption frequency decreases with age"
- **Age Prediction Implication:** Validates the use of alcohol-related questions as strong age predictors in our machine learning models

**If p-value ≥ 0.05:**
- **Decision:** Fail to reject H₀
- **Conclusion:** "No statistically significant evidence of an age-alcohol relationship"
- **Age Prediction Implication:** Questions the importance of alcohol features in our age prediction models

---

## Statistical Methodology

### Chi-Square Test of Independence
**Purpose:** Tests whether two categorical variables (age group and drinking frequency) are independent

**Assumptions:**
1. Data are from a random sample ✓ (OkCupid users)
2. Expected frequency ≥ 5 in each cell ✓ (large sample size)
3. Observations are independent ✓ (different users)

### Effect Size Measurement
**Cramér's V:** Measures the strength of association between age and drinking
- V = √(χ² / (n × min(rows-1, cols-1)))
- Range: 0 (no association) to 1 (perfect association)
- Interpretation: 0.1 = small, 0.3 = medium, 0.5 = large effect

---

## Connection to Age Prediction Project

### Machine Learning Validation
This hypothesis test serves multiple purposes for our age prediction models:

1. **Feature Validation:** Confirms whether alcohol consumption is indeed a strong age predictor
2. **Model Interpretation:** Provides interpretable evidence for neural network learned patterns  
3. **Feature Engineering:** Informs how to weight alcohol-related questions in future models
4. **Academic Rigor:** Adds statistical foundation to the machine learning approach

### Expected Outcomes
- **Strong statistical significance (p < 0.001):** Validates alcohol as a key age predictor
- **Large effect size (Cramér's V > 0.3):** Suggests alcohol consumption patterns are highly age-dependent
- **Clear age gradient:** Supports the use of ordinal encoding for age groups in ML models

---

## Implementation Notes

### Data Requirements
- User age data (d_age column)
- Alcohol consumption responses (q77 column)  
- Minimum sample size: ~1000 users per age group for reliable statistics

### Visualization Strategy
1. **Contingency Table Heatmap:** Shows observed vs expected frequencies
2. **Stacked Bar Chart:** Displays drinking patterns by age group
3. **Statistical Summary:** P-value, effect size, and confidence intervals

### Quality Assurance
- Cross-validate results with existing ML model feature importance
- Compare patterns with demographic literature on alcohol consumption
- Ensure statistical assumptions are met before drawing conclusions

---

## References and Further Reading

- **Statistical Framework:** Based on lesson 9 "Empirical Frameworks and Experiments"
- **Chi-square Test:** Standard test for categorical variable independence
- **Age Prediction Context:** Validates features used in VoyageAI embedding models
- **OkCupid Dataset:** Kirkegaard and Bjerrekær (2016) public dataset

---

*This analysis complements the machine learning age prediction models by providing statistical evidence for the relationship between age and alcohol consumption patterns among online dating users.*
