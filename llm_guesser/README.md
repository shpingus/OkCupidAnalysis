# LLM Age Guesser

A Python package for predicting user ages from survey responses using Large Language Models and Google's Gemini API.

## Features

- **Data Loading**: Load and filter user survey data with flexible column selection
- **Text Formatting**: Convert raw survey responses into natural language descriptions
- **LLM Prediction**: Use Google Gemini to predict ages with confidence scores and explanations  
- **Pipeline Integration**: Complete end-to-end pipeline with evaluation metrics
- **Batch Processing**: Handle multiple users efficiently with tqdm progress bars and rate limiting
- **Column Selection**: Easily specify which survey questions to include in predictions

## Setup

### 1. Install Dependencies
```bash
pip install google-generativeai pandas python-dotenv numpy tqdm
```

### 2. Set up Google API Key
Create a `.env` file in the project root and add:
```
GOOGLE_API_KEY=your_api_key_here
```

Or set as environment variable:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### 3. Data Files
Ensure you have the required data files:
- `data/user_data_public.csv` - User survey responses
- `data/question_data.csv` - Question metadata and text

## Quick Start

### Basic Usage
```python
from llm_guesser import AgePredictionPipeline

# Initialize pipeline
pipeline = AgePredictionPipeline()

# Predict ages for 10 users
results = pipeline.run_complete_pipeline(
    n_samples=10,
    columns=None,  # Auto-select optimal columns
    max_questions=20,
    save_results=True
)

print(f"Average error: {results['evaluation']['mae']:.1f} years")
```

### Component Usage
```python
from llm_guesser import UserDataLoader, TextFormatter, LLMPredictor

# Load data
loader = UserDataLoader()
features, ages = loader.get_rows(n_samples=5)

# Format to text
formatter = TextFormatter(loader)
texts = formatter.format_batch(features)

# Predict ages
predictor = LLMPredictor()
predictions = predictor.predict_batch(texts)
```

## API Reference

### AgePredictionPipeline
Complete pipeline for age prediction.

**Key Methods:**
- `predict_ages()` - Run predictions with filtering options
- `evaluate_results()` - Calculate accuracy metrics
- `run_complete_pipeline()` - Full pipeline with evaluation and saving

### UserDataLoader
Handles loading and filtering user data.

**Key Methods:**
- `get_rows()` - Load filtered user data
- `get_question_info()` - Get question metadata
- `get_data_summary()` - Dataset statistics

### TextFormatter  
Converts user data to descriptive text.

**Key Methods:**
- `format_user_row()` - Format single user
- `format_batch()` - Format multiple users

### LLMPredictor
Handles Gemini API interactions.

**Key Methods:**
- `predict_age()` - Single prediction
- `predict_batch()` - Batch predictions
- `evaluate_predictions()` - Calculate metrics

## Configuration Options

### Pipeline Parameters
- `n_samples` - Number of users to process
- `columns` - Specific columns to include (None for auto-selection)
- `max_questions` - Questions per user prompt
- `min_age/max_age` - Age range filters
- `include_demographics` - Include demographic info
- `delay_seconds` - API call delay

### Model Options
- `model_name` - Gemini model ("gemini-1.5-flash", "gemini-1.5-pro")
- `temperature` - Response randomness (0.0-1.0)
- `max_output_tokens` - Response length limit

## Output Format

### AgePrediction Object
```python
AgePrediction(
    predicted_age=28.5,
    confidence=0.75,
    explanation="Based on responses indicating...",
    raw_response="Full API response",
    success=True,
    error_message=None
)
```

### Evaluation Metrics
- `mae` - Mean Absolute Error
- `rmse` - Root Mean Square Error  
- `accuracy_5y/10y` - Accuracy within 5/10 years
- `success_rate` - API call success rate
- `mean_confidence` - Average confidence score

## Examples

Run the example script to see the package in action:
```bash
cd llm_guesser
python example_usage.py
```

This will demonstrate:
1. Basic pipeline usage
2. Individual component usage
3. Advanced configuration options

## Testing

Test the pipeline with a small sample:
```bash
python -m llm_guesser.pipeline
```

## Limitations

- Requires Google API key with Gemini access
- API costs apply for each prediction
- Accuracy depends on data quality and question selection
- Rate limits may apply based on API tier

## Support

For issues with:
- **Data loading**: Check file paths and CSV format
- **API errors**: Verify API key and quota
- **Poor accuracy**: Try different model or question selection