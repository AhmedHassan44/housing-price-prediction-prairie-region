# Housing Price Prediction in the Prairie Region

## Description
This project uses machine learning techniques to predict housing prices in the Prairie region of Canada. The project covers various stages, including data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, and model evaluation. The goal is to use historical housing price data to forecast future trends and better understand the housing market.

## Dataset
The dataset contains monthly housing price index data, which includes columns such as **Date**, **Location (GEO)**, **Index Type**, and **Price Index Value (VALUE)**. The data is sourced from reliable government statistics, focusing on the Prairie region of Canada. It includes housing price trends over a period, allowing for analysis and prediction of future prices.

## Methodology

- **Data Preprocessing:** 
    - Cleaned the dataset by converting the **REF_DATE** column to a proper datetime format, ensuring that the data is ready for analysis.
    - Handled missing values (if any) and checked for duplicates or erroneous entries.
    - Encoded any categorical variables that could help in prediction (e.g., **GEO** and **INDEX_TYPE**).

- **Exploratory Data Analysis (EDA):** 
    - Used descriptive statistics to summarize and understand the data distribution, trends, and any outliers.
    - Plotted graphs to visually explore how housing prices have evolved over time and across different regions in the Prairie area.

- **Feature Engineering:** 
    - Transformed the **REF_DATE** into an ordinal feature (numeric form) for use in the regression model, as machine learning models generally require numerical input.
    - Selected features that would be most predictive for the target variable (**VALUE**), such as date and index type.

- **Model Selection:** 
    - Applied **Linear Regression** to predict future housing prices based on the historical data. Linear Regression is a good starting point due to its simplicity and interpretability.
    
- **Model Evaluation:** 
    - Evaluated the model using performance metrics such as **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R-squared (R²)** to assess the accuracy and reliability of the predictions.

## Key Findings

- The model demonstrated an ability to predict future housing price trends with reasonable accuracy.
- The housing prices in the Prairie region have shown a steady upward trend, with some seasonal fluctuations.
- By analyzing the features, we identified that **Date** and **Location** were key contributors to predicting future housing prices.

## Replication Instructions

To replicate this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/housing-price-prediction.git
