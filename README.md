# Overview
This repository includes examples of projects from past work projects. The purpose is to demonstrate the Python code used to conduct research, create forecasts, and perform other data science tasks. Each project in this repository uses publicly available information and the Python library faker. 

Each of the projects includes subfolders for the respective data and outputs. More detailed descriptions of each of the folders are provided below:

The most impactful project, a machine learning model that forecasts student enrollment, can be found at: `demo/nae/enrollment`

# Nord Anglia Education

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
    - Applies Linear Regression to staff-related data for predicting test scores and evaluates model performance with metrics like Mean Squared Error, R-squared, F-statistic, and p-value.
    - Generates a scatter plot of predicted versus actual test scores and saves model results in the nae/cost_efficiency/outputs directory.

# Kauffman Foundation

### ARIMA Model Forecasting New Business Applications 
- **Directory:** `demo/emkf`
  - **Script:** `new_bus_apps.py`
    - Implements an ARIMA model for business application forecasting, conducting cross-validation and providing year-by-year predictions. 
    - Utilizes statistical metrics like RMSE and MAE for model evaluation, and includes functions for data exploration and visualization.

  