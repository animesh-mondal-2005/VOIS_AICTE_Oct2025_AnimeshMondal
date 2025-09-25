import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
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

print("\nDataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

for col in ['price', 'service fee']:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.replace('$', '', regex=False).str.replace(',', '').astype(float)

print(f"\nPrice statistics: Min=${df['price'].min():.2f}, Max=${df['price'].max():.2f}, Mean=${df['price'].mean():.2f}")

numeric_features = [
    'Construction year',
    'minimum nights',
    'number of reviews',
    'reviews per month',
    'review rate number',
    'calculated host listings count',
    'availability 365'
]

categorical_features = [
    'room type',
    'neighbourhood group',
    'instant_bookable',
    'cancellation_policy'
]

numeric_features = [col for col in numeric_features if col in df.columns]
categorical_features = [col for col in categorical_features if col in df.columns]

print(f"\nUsing numeric features: {numeric_features}")
print(f"Using categorical features: {categorical_features}")

X_numeric = df[numeric_features].copy()

for col in numeric_features:
    if X_numeric[col].isnull().any():
        X_numeric[col].fillna(X_numeric[col].median(), inplace=True)

label_encoders = {}
X_categorical = pd.DataFrame()

for col in categorical_features:
    if col in df.columns:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        filled_data = df[col].fillna(mode_val)

        le = LabelEncoder()
        X_categorical[col] = le.fit_transform(filled_data)
        label_encoders[col] = le

X_final = pd.concat([X_numeric, X_categorical], axis=1)

y = df['price'].copy()

if y.isnull().any():
    y.fillna(y.median(), inplace=True)

print(f"\nFinal feature set: {X_final.columns.tolist()}")
print(f"Training data shape: {X_final.shape}")

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_mse = mean_squared_error(y_train, train_preds)
test_mse = mean_squared_error(y_test, test_preds)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, test_preds)
r2 = r2_score(y_test, test_preds)

print("\nModel Performance:")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"R² Score: {r2:.4f}")

feature_importance = pd.DataFrame({
    'feature': X_final.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

print("\nCreating Test Cases...")

room_types = df['room type'].value_counts().index[:2].tolist()
neighbourhoods = df['neighbourhood group'].value_counts().index[:2].tolist()

test_cases_data = []

test_case_1 = {}
for feature in X_final.columns:
    if feature in numeric_features:
        if feature == 'Construction year':
            test_case_1[feature] = 2018
        elif feature == 'minimum nights':
            test_case_1[feature] = 2
        elif feature == 'number of reviews':
            test_case_1[feature] = 25
        elif feature == 'reviews per month':
            test_case_1[feature] = 2.5
        elif feature == 'review rate number':
            test_case_1[feature] = 4.8
        elif feature == 'calculated host listings count':
            test_case_1[feature] = 2
        elif feature == 'availability 365':
            test_case_1[feature] = 180
        else:
            test_case_1[feature] = X_final[feature].median()
    else:
        if feature == 'room type':
            test_case_1[feature] = label_encoders[feature].transform(['Entire home/apt'])[0]
        elif feature == 'neighbourhood group':
            test_case_1[feature] = label_encoders[feature].transform([neighbourhoods[0]])[0]
        elif feature == 'instant_bookable':
            test_case_1[feature] = 1  # Assuming 1 = True
        elif feature == 'cancellation_policy':
            test_case_1[feature] = label_encoders[feature].transform(['flexible'])[0]

test_case_2 = test_case_1.copy()
test_case_2['room type'] = label_encoders['room type'].transform(['Private room'])[0]
test_case_2['neighbourhood group'] = label_encoders['neighbourhood group'].transform([neighbourhoods[1]])[0]
test_case_2['review rate number'] = 4.5
test_case_2['number of reviews'] = 15

test_cases = pd.DataFrame([test_case_1, test_case_2])

test_cases = test_cases[X_final.columns]

print("\nTest cases being used:")
print(test_cases)

predicted_prices = model.predict(test_cases)

print("\nTest Case Predictions:")
for i, price in enumerate(predicted_prices, 1):
    print(f"Case {i}: Predicted Price = ${price:.2f}")

print("\nDataset Statistics for Comparison:")
print(f"Minimum price: ${df['price'].min():.2f}")
print(f"Maximum price: ${df['price'].max():.2f}")
print(f"Average price: ${df['price'].mean():.2f}")
print(f"Median price: ${df['price'].median():.2f}")

print("\nSample Predictions vs Actual (from test set):")
sample_results = pd.DataFrame({
    'Actual': y_test.values[:5],
    'Predicted': test_preds[:5],
    'Difference': np.abs(y_test.values[:5] - test_preds[:5])
})
sample_results.index = range(1, 6)
print(sample_results)