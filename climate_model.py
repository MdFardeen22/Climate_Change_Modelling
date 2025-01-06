import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the historical temperature data
data = pd.read_csv('historical_temperature_data.csv')

# Inspect the data
print(data.head())

# Check for missing values
print(data.isna().sum())

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
data[['Temperature']] = imputer.fit_transform(data[['Temperature']])

# Prepare the data for modeling
X = data[['Year']]
y = data['Temperature']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate the models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {"MSE": mse, "R2": r2}
    print(f"{model_name} - Mean Squared Error: {mse}, R^2 Score: {r2}")

# Visualize the results
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', marker='x', label='Predicted')
    plt.xlabel('Year (Scaled)')
    plt.ylabel('Temperature')
    plt.title(f'Temperature Prediction using {model_name}')
    plt.legend()
    plt.show()

# Predict future temperatures
future_years = np.array([[2025], [2030], [2035], [2040], [2045], [2050]])
future_years_scaled = scaler.transform(future_years)
future_predictions = {model_name: model.predict(future_years_scaled) for model_name, model in models.items()}

# Visualize future predictions
for model_name, future_temps in future_predictions.items():
    plt.figure(figsize=(10, 6))
    plt.plot(future_years.flatten(), future_temps, marker='o', label=f'{model_name} Predictions')
    plt.xlabel('Year')
    plt.ylabel('Predicted Temperature')
    plt.title(f'Future Temperature Predictions using {model_name}')
    plt.legend()
    plt.show()

print(f"Predicted temperatures for future years: {future_predictions}")