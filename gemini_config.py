#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gemini API Configuration

Contains configuration settings for the Gemini-based age prediction system.
Copy this file and set your API key in the GEMINI_API_KEY variable below.
"""

from dotenv import load_env
import os
# ================================
# API CONFIGURATION
# ================================

# TODO: Replace with your actual Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# TODO: Optional - Replace with your VoyageAI API key if comparing with existing models
VOYAGE_API_KEY = "your_voyage_api_key_here"

# ================================
# MODEL CONFIGURATION
# ================================

# Gemini model to use
# Options: "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"
GEMINI_MODEL = "gemini-1.5-flash"

# Generation parameters
TEMPERATURE = 0.1  # Lower for more deterministic outputs
MAX_OUTPUT_TOKENS = 100
TOP_P = 0.8
TOP_K = 40

# ================================
# DATA CONFIGURATION
# ================================

# Default number of users to sample for testing
DEFAULT_SAMPLE_SIZE = 1000

# Maximum number of questions to include in a single prompt
MAX_QUESTIONS_PER_PROMPT = 50

# Batch size for API calls (to avoid rate limits)
BATCH_SIZE = 10

# ================================
# PREDICTION CONFIGURATION
# ================================

# Default target to predict
DEFAULT_TARGET = "d_age"

# Alternative targets you can experiment with:
ALTERNATIVE_TARGETS = [
    "d_age",  # Age (default)
    "d_gender",  # Gender
    "d_orientation",  # Sexual orientation
    "d_education_type",  # Education level
    "d_income",  # Income level
    "d_job",  # Job/occupation
]

# ================================
# FILE PATHS
# ================================

# Data file paths
DATA_PATH = "data/user_data_public.csv"
QUESTION_DATA_PATH = "data/question_data.csv"
TEST_ITEMS_PATH = "data/test_items.csv"

# Results directory
RESULTS_DIR = "gemini_results"

# Model comparison files
NN_MODEL_PATHS = [
    "best_age_model.pth",
    "best_simple_model.pth"
]
