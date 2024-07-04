import pandas as pd
import numpy as np

alcohol_sales_df = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/Alcohol_Sales (1).csv")
miles_traveled_df = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/Miles_Traveled (1).csv")

null_values_alcohol = alcohol_sales_df.isnull().sum()
print("Null values in Alcohol Sales dataset:\n", null_values_alcohol)

# Check for null values in miles_traveled_df
null_values_miles = miles_traveled_df.isnull().sum()
print("Null values in Miles Traveled dataset:\n", null_values_miles)

# Display the first few rows of alcohol_sales_df
print("Head of Alcohol Sales dataset:\n", alcohol_sales_df.head())

# Display the last few rows of alcohol_sales_df
print("Tail of Alcohol Sales dataset:\n", alcohol_sales_df.tail())

# Display the first few rows of miles_traveled_df
print("Head of Miles Traveled dataset:\n", miles_traveled_df.head())

# Display the last few rows of miles_traveled_df
print("Tail of Miles Traveled dataset:\n", miles_traveled_df.tail())


# Describe the statistical summary of alcohol_sales_df
describe_alcohol = alcohol_sales_df.describe()
print("Description of Alcohol Sales dataset:\n", describe_alcohol)

# Describe the statistical summary of miles_traveled_df
describe_miles = miles_traveled_df.describe()
print("Description of Miles Traveled dataset:\n", describe_miles)


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of Alcohol Sales
plt.figure(figsize=(12, 6))
sns.histplot(alcohol_sales_df['S4248SM144NCEN'], kde=True, bins=30)
plt.title('Distribution of Alcohol Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of Miles Traveled
plt.figure(figsize=(12, 6))
sns.histplot(miles_traveled_df['TRFVOLUSM227NFWA'], kde=True, bins=30)
plt.title('Distribution of Miles Traveled')
plt.xlabel('Miles Traveled')
plt.ylabel('Frequency')
plt.show()

# Convert DATE columns to datetime objects and set them as index
alcohol_sales_df['DATE'] = pd.to_datetime(alcohol_sales_df['DATE'], format='%d-%m-%Y')
miles_traveled_df['DATE'] = pd.to_datetime(miles_traveled_df['DATE'], format='%d-%m-%Y')

alcohol_sales_df.set_index('DATE', inplace=True)
miles_traveled_df.set_index('DATE', inplace=True)

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Function to forecast and evaluate using ARIMA
def forecast_arima(train, test, column_name, order=(5, 1, 0)):
    # Train the ARIMA model
    arima_model = ARIMA(train, order=order)
    arima_model_fit = arima_model.fit()

    # Forecast future values
    forecast = arima_model_fit.forecast(steps=len(test))
    test['Forecast'] = forecast

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(test[column_name], test['Forecast']))
    mae = mean_absolute_error(test[column_name], test['Forecast'])

    return test, rmse, mae

# Split the Alcohol Sales data
train_size_alcohol = int(len(alcohol_sales_df) * 0.8)
train_alcohol, test_alcohol = alcohol_sales_df[:train_size_alcohol], alcohol_sales_df[train_size_alcohol:]

# Split the Miles Traveled data
train_size_miles = int(len(miles_traveled_df) * 0.8)
train_miles, test_miles = miles_traveled_df[:train_size_miles], miles_traveled_df[train_size_miles:]

# Forecast and evaluate for Alcohol Sales
test_alcohol, rmse_alcohol, mae_alcohol = forecast_arima(train_alcohol, test_alcohol, 'S4248SM144NCEN')

# Forecast and evaluate for Miles Traveled
test_miles, rmse_miles, mae_miles = forecast_arima(train_miles, test_miles, 'TRFVOLUSM227NFWA')

print("Alcohol Sales - RMSE:", rmse_alcohol, "MAE:", mae_alcohol)
print("Miles Traveled - RMSE:", rmse_miles, "MAE:", mae_miles)


# Visualize the forecast results for Alcohol Sales
plt.figure(figsize=(12, 6))
plt.plot(train_alcohol['S4248SM144NCEN'], label='Train')
plt.plot(test_alcohol['S4248SM144NCEN'], label='Test')
plt.plot(test_alcohol['Forecast'], label='Forecast')
plt.legend()
plt.title('Alcohol Sales Forecasting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Visualize the forecast results for Miles Traveled
plt.figure(figsize=(12, 6))
plt.plot(train_miles['TRFVOLUSM227NFWA'], label='Train')
plt.plot(test_miles['TRFVOLUSM227NFWA'], label='Test')
plt.plot(test_miles['Forecast'], label='Forecast')
plt.legend()
plt.title('Miles Traveled Forecasting')
plt.xlabel('Date')
plt.ylabel('Miles Traveled')
plt.show()

# Bar graph of Alcohol Sales by month
plt.figure(figsize=(12, 6))
alcohol_sales_df['Month'] = alcohol_sales_df.index.month
monthly_sales = alcohol_sales_df.groupby('Month')['S4248SM144NCEN'].sum()
monthly_sales.plot(kind='bar')
plt.title('Monthly Alcohol Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()


# Pie chart of total sales per year for Alcohol Sales
plt.figure(figsize=(12, 6))
alcohol_sales_df['Year'] = alcohol_sales_df.index.year
yearly_sales = alcohol_sales_df.groupby('Year')['S4248SM144NCEN'].sum()
yearly_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title('Yearly Alcohol Sales')
plt.ylabel('')
plt.show()


# Histogram of Miles Traveled
plt.figure(figsize=(12, 6))
plt.hist(miles_traveled_df['TRFVOLUSM227NFWA'], bins=30)
plt.title('Histogram of Miles Traveled')
plt.xlabel('Miles Traveled')
plt.ylabel('Frequency')
plt.show()


