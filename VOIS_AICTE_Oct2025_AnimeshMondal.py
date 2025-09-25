import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from google.colab import files

uploaded = files.upload()
file_path = list(uploaded.keys())[0]

if file_path.endswith('.csv'):
    df = pd.read_csv(file_path)
elif file_path.endswith(('.xls', '.xlsx')):
    df = pd.read_excel(file_path)
else:
    raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

print("Data Preview:")
print(df.head())

print("Available columns in dataset:")
print(df.columns.tolist())

features = ["Bedrooms", "Bathrooms", "Cleanliness_rating", "Accuracy_rating", "Communication_rating"]
features = [col for col in features if col in df.columns]

X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Mean Squared Error: {mse:.2f}")

test_cases = pd.DataFrame([
    {"Bedrooms": 2, "Bathrooms": 1, "Cleanliness_rating": 9, "Accuracy_rating": 8, "Communication_rating": 9},
    {"Bedrooms": 3, "Bathrooms": 2, "Cleanliness_rating": 10, "Accuracy_rating": 9, "Communication_rating": 10}
])

test_cases = test_cases[[col for col in features if col in test_cases.columns]]

predicted_prices = model.predict(test_cases)

print("\nTest Case Predictions:")
for i, price in enumerate(predicted_prices, 1):
    print(f"Case {i}: Predicted Price = ${price:.2f}")