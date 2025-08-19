# OkCupid Age Prediction Analysis

## 1. Project Overview

The OkCupid Age Prediction project aims to develop machine learning models that can accurately predict a user's age based on their demographic information and question responses. This project explores four different model architectures with varying levels of complexity to determine the most effective approach for this task.

The primary goals of this project are:
- To evaluate whether embedding-based approaches can effectively capture relationships in categorical and limited textual data
- To compare simple vs. ensemble modeling techniques
- To assess whether enhanced feature engineering techniques improve prediction accuracy
- To determine the most effective model architecture for age prediction

## 2. Data Preprocessing

**all models use the same core preprocessing approach**. The preprocessing includes:

1. **Demographic data preparation**: Collecting fields starting with 'd_' (e.g., d_body_type, d_drinks, d_education) and combining them into a single text field called 'combined_demographics'.

2. **Question response preparation**: Selecting high-response questions (with >50,000 responses) and combining their answers into a text field called 'question_responses'. The number of responses per question is determined from the 'N' column in the question_data.csv file, which contains question metadata.

3. **Basic text cleaning**: Lowercasing all text and normalizing whitespace.


## 3. VoyageAI Embeddings

Embeddings are dense vector representations of data that capture semantic meaning and relationships. They transform categorical or textual data into continuous vector spaces where semantically similar items are positioned closer together.

VoyageAI is a specialized AI company offering state-of-the-art text embedding models that excel at capturing semantic relationships in various types of text data; we chose them for this project because their embeddings are particularly effective at preserving subtle semantic similarities in short text snippets, making them ideal for our demographic data and question responses.

### Benefits for Our Project

Using VoyageAI embeddings in this project offers several key advantages:

1. **Handling categorical data efficiently**: Instead of using traditional one-hot encoding for demographic variables (which would create sparse high-dimensional vectors), embeddings provide a compact representation while preserving semantic relationships.

2. **Capturing subtle relationships**: VoyageAI embeddings understand that values like "occasionally" and "sometimes" for drinking habits are more similar than "never" and "frequently" - nuances that traditional encoding methods might miss.

3. **Leveraging pre-trained knowledge**: Despite our limited textual content, the embeddings bring pre-trained knowledge about language relationships that helps understand the connections between demographic values and question responses better than training from scratch.

4. **Enabling advanced feature engineering**: Our enhanced models compute element-wise products and differences between embeddings, capturing complex interactions between demographic and question data.

## 4. Model Architectures

Our project explores four model architectures, each building upon the previous one to improve performance:

### Simple Model
- **Architecture**: Basic 3-layer neural network with ReLU activations and dropout (0.2, 0.1)
- **Input**: Concatenated embeddings from demographic data and question responses
- **Why this design**: We chose a simple feedforward architecture as our start point to establish performance benchmarks. The moderate dropout rates (0.2, 0.1) provide regularization without being too aggressive, as we expected the embedding features to already be meaningful.
- **Prediction based on**: User demographic information (d_body_type, d_diet, d_drinks, etc.) and responses to high-engagement questions

### K-Fold Ensemble Model
- **Architecture**: Same 3-layer neural network as the Simple Model, but trained using 5-fold cross-validation
- **Key enhancement**: Creates 5 separate models on different data splits and averages their predictions
- **Why K-fold is beneficial**: 
  - Provides more stable predictions by averaging results from multiple models
  - Increases model robustness through training on different data distributions
  - Provides a more reliable performance estimate by using all data for both training and validation
  - Helps mitigate the impact of lucky/unlucky data splits
- **Prediction based on**: Same features as the Simple Model

### Enhanced Simple Model
- **Architecture**: Modified 3-layer neural network with stronger dropout (0.5, 0.3) and Xavier uniform weight initialization
- **Key enhancements over Simple Model**:
  1. **Advanced feature engineering**: Creates element-wise products and differences between embedding vectors to capture feature interactions
  2. **Stronger regularization**: Higher dropout rates (0.5, 0.3) to prevent overfitting on the more complex feature space
  3. **Better weight initialization**: Xavier uniform initialization for more stable training, compared to the Kaiming normal initialization used in the simple model. While Kaiming is well-suited for ReLU networks, Xavier helps maintain consistent gradient scales across layers, which is beneficial when working with complex feature interactions.

- **Why these enhancements**: 
  - Feature interactions often capture important relationships that simple concatenation misses.
  - Stronger regularization helps manage the increased model capacity needed to learn from the enhanced features
- **Prediction based on**: Same base features as Simple Model, but with additional engineered interaction features

### Enhanced K-Fold Model
- **Architecture**: Combines the enhanced architecture with 5-fold cross-validation ensemble
- **Key enhancement**: Merges the benefits of both the Enhanced Simple Model and the K-Fold approach
- **Why this combination**: Theoretical expectation that combining advanced feature engineering with ensemble methods would yield the best performance
- **Prediction based on**: Same enhanced features as the Enhanced Simple Model

### Advanced Feature Engineering Details

The enhanced models use specific mathematical operations to create richer feature representations:

1. **Element-wise Products**: We compute the element-wise multiplication between demographic and question embeddings:
   ```python
   demo_question_product = demographic_embeddings_np[:, :min_dim] * question_embeddings_np[:, :min_dim]
   ```
   This operation multiplies each dimension of the demographic embedding vector with the corresponding dimension of the question response embedding vector.

2. **Element-wise Differences**: We calculate the difference between each dimension:
   ```python
   demo_question_diff = demographic_embeddings_np[:, :min_dim] - question_embeddings_np[:, :min_dim]
   ```
   This captures the "distance" or discrepancy between how demographic information and question responses are represented in the embedding space.

The intuition behind this approach is:
- The element-wise product captures when both embeddings have strong signals in the same dimension, highlighting where demographic and question response data agree or reinforce each other
- The element-wise difference captures disparities between demographic and question response information, potentially revealing contradictions or unique information that's only present in one data source

These operations effectively enrich the feature space, providing the model with much deeper information about relationships between different data types, which helps explain the improved R² score observed in the enhanced models.

## 5. Results Analysis

According to our model comparison report, all four models achieved relatively close results, with MAE scores ranging from 5.20 to 5.23 years. Similarly, the accuracy within ±5 years is remarkably consistent, ranging from 59.8% to 61.1% across all models, suggesting a fundamental limit to prediction precision with the available features. Here's how they compare:

| Model | MAE (years) ↓ | R² Score ↑ | ±1 year (%) ↑ | ±3 years (%) ↑ | ±5 years (%) ↑ |
| ----- | ------------ | ---------- | ------------- | -------------- | -------------- |
| Simple Model | 5.22 | 0.1649 | 15.5% | 39.9% | 61.1% |
| Kfold Model | 5.23 | 0.1613 | 15.0% | 38.9% | 60.8% |
| Enhanced Simple Model | 5.20 | 0.1843 | 14.1% | 39.6% | 59.8% |
| Enhanced Kfold Model | 5.23 | 0.1661 | 15.5% | 39.7% | 61.0% |

**The Enhanced Simple Model emerges as the best performer overall**, with the lowest MAE (5.20 years) and highest R² score (0.1843). However, the un-enhanced simple Model actually performed better on accuracy within specific year thresholds.

### Surprising Results and Explanations

It's somewhat surprising that the Enhanced Simple Model outperformed the Enhanced K-Fold Model, which theoretically should have combined the advantages of both approaches. This could be explained by:

1. **Diminishing returns from enhanced features**: The element-wise product and difference features may already capture the most important patterns in the data so effectively that the additional complexity of ensemble averaging provides minimal benefit. Once you've captured most of the available signal with sophisticated features, additional model complexity might not translate to better performance.

2. **Training variance**: The difference between models is very small (5.20 vs 5.23 MAE), which could be within the range of random variation from training. With more runs or different random seeds, the results might average out differently.

All models reached very similar results, with MAE differences of less than 0.03 years between the best and worst models. This suggests that for this specific task and dataset, the core embedding approach is capturing most of the available signal, and architectural variations have diminishing returns.

## 6. Project Setup Requirements

To run this project, you'll need:

1. **Python Virtual Environment**:
   ```bash
   # Create virtual environment
   python -m venv okcupid_env
   
   # Activate it (MacOS/Linux)
   source okcupid_env/bin/activate
   # OR Windows
   okcupid_env\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **VoyageAI API Key**:
   - Create a `.env` file in the project root directory
   - Add your API key: `VOYAGE_API_KEY=your_api_key_here`

## 7. Running the Models

### Common Parameters for All Models

All models accept similar command line parameters:

- `--data_path`: Path to the OkCupid user data CSV file (default: 'data/user_data_public.csv')
- `--question_data_path`: Path to the question data CSV file (default: 'data/question_data.csv')
- `--sample_size`: Number of users to sample (default: 1000, recommended: 5000 for better results)
- `--cache_dir`: Directory to cache embeddings (default: 'cache')
- `--embedding_model`: VoyageAI model to use (default: 'voyage-2')
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Maximum number of training epochs (default: 50)
- `--patience`: Early stopping patience (default: 10)

### Running Individual Models

1. **Simple Model**:
   ```bash
   python voyageai_age_simple_model.py --sample_size 5000
   ```
   Results will be saved to the `results/` directory.

2. **K-Fold Model**:
   ```bash
   python voyageai_age_k_fold_model.py --sample_size 5000
   ```
   Results will be saved to the `results/` directory.

3. **Enhanced Simple Model**:
   ```bash
   python voyageai_age_enhanced_simple_model.py --sample_size 5000
   ```
   Results will be saved to the `enhanced_models_results/` directory.

4. **Enhanced K-Fold Model**:
   ```bash
   python voyageai_age_enhanced_k_fold_model.py --sample_size 5000
   ```
   Results will be saved to the `enhanced_models_results/` directory.

## 8. Viewing Results

After running the models, you can view the comparison results:

1. **Open the HTML report**:
   ```bash
   # On macOS
   open model_comparisons/model_comparison_report.html
   
   # On Windows
   start model_comparisons/model_comparison_report.html
   
   # On Linux
   xdg-open model_comparisons/model_comparison_report.html
   ```
   
2. **Alternative: Using Python's HTTP server**:
   ```bash
   python -m http.server
   ```
   Then navigate to `http://localhost:8000/model_comparisons/model_comparison_report.html` in your browser.

3. **View comparison plots directly**:
   ```bash
   open model_comparisons/all_models_comparison.png
   ```

You can also examine the individual model results in their respective output directories (`results/` and `enhanced_models_results/`).

To generate a new comparison report after running models:
```bash
python model_comparison.py
```

## Project Structure

```
OkCupidAnalysis/
├── data/                              # Input data files
│   ├── question_data.csv              # Question metadata
│   └── user_data_public.csv           # User data
├── model_comparisons/                 # Model comparison results
│   ├── all_models_comparison.png      # Overall comparison
│   ├── model_comparison_report.html   # Detailed comparison report
│   ├── model_comparison_report.md     # Markdown version of report
│   └── */                             # Pairwise comparison folders
├── enhanced_models_results/           # Results from enhanced models
├── results/                           # Results from basic models
├── voyageai_age_simple_model.py       # Simple model implementation
├── voyageai_age_k_fold_model.py       # K-fold ensemble model
├── voyageai_age_enhanced_simple_model.py    # Enhanced simple model
├── voyageai_age_enhanced_k_fold_model.py    # Enhanced k-fold model
├── model_comparison.py                # Script for comparing models
└── requirements.txt                   # Package dependencies
