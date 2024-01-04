# Overview
This repository includes examples of projects from past work. The purpose is to demonstrate the Python code used to conduct research, create forecasts, and perform other data science tasks. Each project in this repo uses synthetic data generated from the Python library `faker` and other publicly available information.  

The most impactful project, a machine learning (ML) model that forecasts student enrollment, can be found at: `demo/nae/enrollment`

Each of the projects includes subfolders for the respective data and outputs. More detailed descriptions of each of the projects are provided below:

# Nord Anglia

### ML Enrollment Forecasts
- **Directory:** `demo/nae/enrollment`
  - **Script:** `prediction__model_fit.py`
    - This script handles the entire machine learning pipeline for forecasting student enrollment.
    - It includes data preparation, preprocessing, feature engineering, model training using RandomForestClassifier, and model evaluation using RandomizedSearchCV.

### NLP Parent Survey Categorization and Sentiment Analysis
- **Directory:** `demo/nae/nlp_survey`
  - **Script:** `sentiment_analysis.py`
    - This script demonstrates sentiment analysis on parent survey responses using the transformers library.
    - Synthetic survey data is generated, preprocessed, and sentiment analysis is applied, providing labeled sentiments and scores.

### Staff Cost Efficiency 
- **Directory:** `demo/nae/cost_efficiency`
  - **Script:** `staff_cost_model.py`
    - This script demonstrates how I use ML to create counterfactuals
    - Applies Linear Regression to staff-related data for predicting test scores and evaluates model performance with metrics like Mean Squared Error, R-squared, F-statistic, and p-value.
    - Generates a scatter plot of predicted versus actual test scores and saves model results in the nae/cost_efficiency/outputs directory.

### ANOVA Tests: Writing Scores by Native Language
- **Directory:** `nae/education`
  - **Script:** `writing.py`
    - This script was used to teach a Data Science and Statistics lesson at William Jewell College on November 29, 2023. The purpose of the lesson was to introduce ANOVA and demonstrate how I use ANOVA for business intelligence.
    - The code calculates descriptive statistics, generates histograms, and performs statistical tests, including Shapiro-Wilk for normality, one-way ANOVA, and Tukey's HSD post hoc test.

# Kauffman Foundation

### ARIMA Model Forecasting New Business Applications 
- **Directory:** `demo/emkf`
  - **Script:** `new_bus_apps.py`
    - Implements an ARIMA model for business application forecasting, conducting cross-validation and providing year-by-year predictions. 
    - Utilizes statistical metrics like RMSE and MAE for model evaluation, and includes functions for data exploration and visualization.

# Independent Projects

### ARIMA Model Forecasting New Business Applications 
- **Directory:** `demo/independent`
  - **Script:** `file_mover.py`
    - Demonstrates tools that can be used to work with large data and files 
    - Uses the faker library to generate a large dataset, msgpack library to serialize and deserialize data, and shutil to move the saved .msgpack files between locations.  
    
  