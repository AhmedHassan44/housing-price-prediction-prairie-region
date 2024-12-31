# -------------------------------------------------------
# Inserting Data In Sql from Excel Csv file
# -------------------------------------------------------

'''
# Load the dataset
housing_data = pd.read_csv('/Users/ahmedhasan/Desktop/monthlyHousingPriceIndexClean.csv')

# Add a default day ('01') to each date value to make it 'YYYY-MM-DD' format
housing_data['REF_DATE'] = housing_data['REF_DATE'] + '-01'

# Convert 'REF_DATE' to datetime format
housing_data['REF_DATE'] = pd.to_datetime(housing_data['REF_DATE'], format='%Y-%m-%d')

# Establish MySQL connection
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="17#Mysql17",
    database="housing_analysis"
)
cursor = db_connection.cursor()

# SQL Insert query
insert_query = """
INSERT INTO housing_price_index (REF_DATE, GEO, INDEX_TYPE, VALUE)
VALUES (%s, %s, %s, %s)
"""

# Loop through the DataFrame and insert data into MySQL
for _, row in housing_data.iterrows():
    cursor.execute(insert_query, (row['REF_DATE'], row['GEO'], row['New housing price indexes'], row['VALUE']))

# Commit the changes and close the connection
db_connection.commit()
print("Data inserted successfully!")
cursor.close()
db_connection.close()
'''

# -------------------------------------------------------
# Step 1: Import libraries
# -------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------
# Step 2: Load dataset
# -------------------------------------------------------

# Load the dataset
housing_data = pd.read_csv('/Users/ahmedhasan/Desktop/monthlyHousingPriceIndexClean.csv')

# Check the first few rows to understand the structure of the data
print("First 5 rows of the dataset:")
print(housing_data.head())

# -------------------------------------------------------
# Step 3: Exploratory Data Analysis (EDA)
# -------------------------------------------------------

# Check for missing values
print("\nMissing values in each column:")
print(housing_data.isnull().sum())

# Summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(housing_data.describe())

# -------------------------------------------------------
# Step 4: Feature Engineering
# -------------------------------------------------------

# Convert 'REF_DATE' to ordinal (numeric) for regression
housing_data['REF_DATE'] = pd.to_datetime(housing_data['REF_DATE'])
housing_data['Date_Ordinal'] = housing_data['REF_DATE'].apply(lambda x: x.toordinal())

# Select features (Date_Ordinal) and target (VALUE)
X = housing_data[['Date_Ordinal']]
y = housing_data['VALUE']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# Step 5: Training the Linear Regression Model
# -------------------------------------------------------

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the model using the training data
lr_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr_model.predict(X_test)

# -------------------------------------------------------
# Step 6: Model Evaluation
# -------------------------------------------------------

# Calculate the Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) for the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Linear Regression - MSE: {mse}')
print(f'Linear Regression - MAE: {mae}')
print(f'Linear Regression - R2: {r2}')

# -------------------------------------------------------
# Step 7: Visualizing the Results
# -------------------------------------------------------

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))

# Scatter plot of true vs predicted values
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')

# Plot the line of perfect prediction (45-degree line)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.show()

# -------------------------------------------------------
# Step 8: Predicting for the Next 5 Years
# -------------------------------------------------------

import numpy as np

# Get the last date in the dataset and create future dates
last_date = housing_data['REF_DATE'].max()  # Get the most recent date in the dataset

# Generate the next 60 months (5 years * 12 months)
future_dates = pd.date_range(start=last_date, periods=60, freq='M')  # 60 months = 5 years

# Convert the future dates to ordinal format (same as before)
future_dates_ordinal = future_dates.to_series().apply(lambda x: x.toordinal())

# Reshape the data to match the input shape for prediction (2D array)
future_dates_ordinal = future_dates_ordinal.values.reshape(-1, 1)

# Make predictions for the future dates
future_predictions = lr_model.predict(future_dates_ordinal)

# Create a DataFrame for the future predictions
future_predictions_df = pd.DataFrame({
    'REF_DATE': future_dates,
    'Date_Ordinal': future_dates_ordinal.flatten(),
    'Predicted_VALUE': future_predictions
})

# Display the future predictions
print(future_predictions_df)

# -------------------------------------------------------
# Step 9: Visualizing the Predictions for the Next 5 Years
# -------------------------------------------------------

# Plot the past and future predictions
plt.figure(figsize=(12, 6))

# Plot the original data
plt.plot(housing_data['REF_DATE'], housing_data['VALUE'], label='Historical Data', color='blue')

# Plot the future predictions
plt.plot(future_predictions_df['REF_DATE'], future_predictions_df['Predicted_VALUE'], label='Predictions for Next 5 Years', color='red', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Housing Price Index (VALUE)')
plt.title('Housing Price Index: Historical Data and Predictions for the Next 5 Years')
plt.legend()
plt.xticks(rotation=45)
plt.show()
