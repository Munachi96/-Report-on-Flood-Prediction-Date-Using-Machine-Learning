```python

import pandas as pd

# Load the dataset
file_path = 'latest lagos weather dataset.csv'
lagos_weather_data = pd.read_csv(file_path)

# Display the first few rows and the column names of the dataset
lagos_weather_data.head(), lagos_weather_data.columns

* importing historical flood dataset

import pandas as pd

# Load the datasets
weather_data_path = 'latest lagos weather dataset.csv'
flood_data_path = 'historica_flood.csv'

# Read the CSV files
weather_data = pd.read_csv(weather_data_path)
flood_data = pd.read_csv(flood_data_path)

# Display the first few rows of each dataset
weather_data_head = weather_data.head()
flood_data_head = flood_data.head()

weather_data_head, flood_data_head

# Convert 'datetime' to datetime format
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])

# Check for missing values in both datasets
weather_missing = weather_data.isnull().sum()
flood_missing = flood_data.isnull().sum()

weather_data.dtypes, weather_missing, flood_data.dtypes, flood_missing

# Handle missing values
weather_data['preciptype'].fillna('Unknown', inplace=True)
weather_data['visibility'].fillna(weather_data['visibility'].mean(), inplace=True)
weather_data['severerisk'].fillna(weather_data['severerisk'].mean(), inplace=True)

# Convert 'sunrise' and 'sunset' to datetime if needed
weather_data['sunrise'] = pd.to_datetime(weather_data['sunrise'], errors='coerce')
weather_data['sunset'] = pd.to_datetime(weather_data['sunset'], errors='coerce')

# Verify the changes
weather_data.info()


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting historical flood incidents

plt.figure(figsize=(10, 6))
sns.countplot(x='Year', data=flood_data)
plt.title('Historical Flood Incidents in Lagos')
plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## Observations

#There appears to be variability in flood incidents across different years. 
#Some years show higher numbers of flood incidents than others, 
#suggesting temporal patterns that could potentially be captured by time-based features in machine learning models.


# Plotting precipitation trends

plt.figure(figsize=(10, 6))
plt.plot(weather_data['datetime'], weather_data['precip'], marker='o', linestyle='-', color='b', label='Precipitation')
plt.title('Daily Precipitation Trends in Lagos')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Observations
#Heavy precipitation tends to coincide with increased flood incidents. 
#This suggests that rainfall data, particularly heavy rainfall events, can be a significant predictor for flood prediction models.
#Higher humidity levels might contribute to the likelihood of flood incidents, especially when combined with heavy precipitation. 
#This correlation can be explored further in feature engineering.
#Temperature trends, both maximum and minimum, could indirectly influence flood incidents through their impact on evaporation rates and soil saturation levels.


# Plotting humidity and cloud cover trends

plt.figure(figsize=(12, 6))
plt.plot(weather_data['datetime'], weather_data['humidity'], marker='o', linestyle='-', color='g', label='Humidity')
plt.plot(weather_data['datetime'], weather_data['cloudcover'], marker='o', linestyle='-', color='r', label='Cloud Cover')
plt.title('Humidity and Cloud Cover Trends in Lagos')
plt.xlabel('Date')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

#Observations
#Seasonal variations in weather and flood incidents are evident. 
#Models could benefit from incorporating seasonal indicators or time-series features that capture these recurring patterns.


# Calculate correlation matrix
weather_corr = weather_data[['precip', 'humidity', 'cloudcover', 'windspeed', 'sealevelpressure']].corr()

# Plotting correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(weather_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap: Weather Variables vs. Flood Incidents')
plt.tight_layout()
plt.show()

#Observations

#The correlation heatmap showed potential relationships between weather variables (precipitation, humidity, temperature) and flood incidents. 
#Positive correlations between these variables indicate that they can be predictive features in machine learning models.

df = pd.read_csv('merged_dataset.csv')
df.head()
df.columns


import pandas as pd
import matplotlib.pyplot as plt

# Replacing 'my_dataset.csv' with the path to my dataset file

file_path = 'merged_dataset.csv'
data = pd.read_csv(file_path)

# Converting 'datetime' to datetime object with dayfirst=True
data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True)

# Plot Sea Level Pressure over time
plt.figure(figsize=(12, 6))
plt.plot(data['datetime'], data['Sealevelpressure'], label='Sea Level Pressure')
plt.xlabel('Date')
plt.ylabel('Sea Level Pressure (hPa)')
plt.title('Sea Level Pressure over Time')
plt.legend()
plt.show()

#Observations

# The sea level pressure shows noticeable fluctuations over time.
# Periodic patterns may indicate seasonal changes in atmospheric pressure.


# Plot Windspeed over time
plt.figure(figsize=(12, 6))
plt.plot(data['datetime'], data['Windspeed'], label='Windspeed')
plt.xlabel('Date')
plt.ylabel('Windspeed (km/h)')
plt.title('Windspeed over Time')
plt.legend()
plt.show()

#Observation
 
#The windspeed data exhibits variability, with some periods experiencing higher wind speeds.
#Peaks in windspeed could suggest strong weather systems that might bring precipitation.


# Plot Wind Direction over time
plt.figure(figsize=(12, 6))
plt.plot(data['datetime'], data['Winddir'], label='Wind Direction')
plt.xlabel('Date')
plt.ylabel('Wind Direction (degrees)')
plt.title('Wind Direction over Time')
plt.legend()
plt.show()

#Observations

#The wind direction graph shows the predominant wind directions over time.
#Periods with consistent wind directions might indicate the prevailing winds that carry moisture.


# Plot Precipitation over time

plt.figure(figsize=(12, 6))
plt.plot(data['datetime'], data['Precip'], label='Precipitation')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.title('Precipitation over Time')
plt.legend()
plt.show()

Observations

#The precipitation data reveals the frequency and intensity of rainfall over time.
#Noticeable peaks in precipitation could be indicative of heavy rainfall events that might lead to flooding.



# Assume variations based on historical data for future predictions
future_data['Precip'] = data['Precip'].rolling(window=30, min_periods=1).mean().iloc[-1]
future_data['Sealevelpressure'] = data['Sealevelpressure'].rolling(window=30, min_periods=1).mean().iloc[-1]
future_data['Windspeed'] = data['Windspeed'].rolling(window=30, min_periods=1).mean().iloc[-1]
future_data['river discharge'] = data['river discharge'].rolling(window=30, min_periods=1).mean().iloc[-1]

# Alternatively, you can use the last observed value
# future_data['Precip'] = data['Precip'].iloc[-1]
# future_data['Sealevelpressure'] = data['Sealevelpressure'].iloc[-1]
# future_data['Windspeed'] = data['Windspeed'].iloc[-1]
# future_data['river discharge'] = data['river discharge'].iloc[-1]

# Add other relevant features and lag features as before
future_data['river_discharge_lag1'] = future_data['river discharge'].shift(1).fillna(data['river discharge'].mean())
future_data['Precip_lag1'] = future_data['Precip'].shift(1).fillna(data['Precip'].mean())
future_data['Windspeed_lag1'] = future_data['Windspeed'].shift(1).fillna(data['Windspeed'].mean())

# Ensure future_data DataFrame is correctly prepared
print(future_data.head())


import numpy as np

# Step 1: Define the threshold for precipitation
threshold_precip = 50  # Hypothetical threshold in millimeters

# Step 2: Create the 'Flood' column based on precipitation threshold
data['Flood'] = np.where(data['Precip'] >= threshold_precip, 1, 0)

# Step 3: Verify and display the first few rows
print(data[['datetime', 'Precip', 'Flood']].head())



# Select features (X) and target variable (y)
features = ['import pandas as pd']

# Assuming your dataset is already loaded into 'data'
# Select the columns used in your training set
columns_used = ['river discharge', 'day', 'month', 'year', 'Tempmax', 'Tempmin', 'Temp', 
                'Humidity', 'Precip', 'Preciprob', 'Precipcover', 'Windspeed', 
                'Winddir', 'Sealevelpressure', 'Cloudcover', 'Severrisk', 'Moonphase']

# Calculate summary statistics
summary_stats = data[columns_used].describe().transpose()

# Display summary statistics
print(summary_stats)
]
target = 'Flood'

X = data[features]
y = data[target]

# Display the shapes to verify
print("Features (X) shape:", X.shape)
print("Target variable (y) shape:", y.shape)



from sklearn.model_selection import train_test_split

# Split data into training and test sets (adjust test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes to verify
print("Training set shapes:", X_train.shape, y_train.shape)
print("Test set shapes:", X_test.shape, y_test.shape)


from sklearn.linear_model import LogisticRegression

# Initialize logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# Predict the likelihood of future floods
data['Flood_Prediction'] = model.predict(X)

# Filter the dataset for predicted flood occurrences
predicted_floods = data[data['Flood_Prediction'] == 1]

# Display the predicted dates of future floods
print(predicted_floods[['datetime', 'Flood_Prediction']])


# Plot predicted flood occurrences
plt.figure(figsize=(12, 6))
plt.plot(data['datetime'], data['Flood_Prediction'], label='Predicted Flood Occurrence', linestyle='--', marker='o')
plt.xlabel('Date')
plt.ylabel('Flood Prediction')
plt.title('Predicted Flood Occurrences over Time')
plt.legend()
plt.show()


import pandas as pd

# Assuming you want to predict for the next 30 days
future_dates = pd.date_range(start='2024-07-04', periods=30)

# Create a DataFrame for future predictions
future_data = pd.DataFrame({'datetime': future_dates})

# Add other relevant columns with hypothetical values
# Example:
future_data['Precip'] = 6.76  # Example hypothetical value
future_data['Sealevelpressure'] = 1012.13  # Example hypothetical value
future_data['Windspeed'] = 26.24  # Example hypothetical value
future_data['river discharge'] = 293.63  # Example hypothetical value
future_data['day'] = 15.82
future_data['month'] = 6.22
future_data['year'] = 2023.0
future_data['Tempmax'] = 31.57
future_data['Tempmin'] = 24.43
future_data['Temp'] = 27.65
future_data['Humidity'] = 83.46
future_data['Precip'] = 6.76
future_data['Preciprob'] = 66.06
future_data['Precipcover'] = 5.88
future_data['Winddir'] = 212.55
future_data['Cloudcover'] = 57.48
future_data['Severrisk'] = 46.76
future_data['Moonphase'] = 0.48

# Ensure all necessary columns are present and in the right format
# Example: future_data = future_data[['datetime', 'Precip', 'Sealevelpressure', 'Windspeed', 'river discharge']]


# Predict the likelihood of future floods
future_predictions = model.predict(future_data[['river discharge', 'day', 'month', 'year', 'Tempmax', 'Tempmin', 'Temp', 
                'Humidity', 'Precip', 'Preciprob', 'Precipcover', 'Windspeed', 'Winddir', 'Sealevelpressure', 'Cloudcover', 'Severrisk', 'Moonphase']])


# Add predictions to the future_data DataFrame
future_data['Flood_Prediction'] = future_predictions

# Display the predicted dates of future floods
print(future_data[['datetime', 'Flood_Prediction']])


```
