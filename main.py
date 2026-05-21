import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("car data.csv")

# ==============================
# SHOW DATASET
# ==============================

print("FIRST 5 ROWS OF DATASET:\n")
print(df.head())

print("\nDATASET INFORMATION:\n")
print(df.info())

print("\nMISSING VALUES:\n")
print(df.isnull().sum())

# ==============================
# CONVERT TEXT DATA INTO NUMBERS
# ==============================

le = LabelEncoder()

df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])

df['Selling_type'] = le.fit_transform(df['Selling_type'])

df['Transmission'] = le.fit_transform(df['Transmission'])

# ==============================
# FEATURE ENGINEERING
# ==============================

# Create new column: Car Age
df['Car_Age'] = 2026 - df['Year']

# Remove unnecessary columns
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# ==============================
# FEATURES AND TARGET
# ==============================

X = df.drop('Selling_Price', axis=1)

y = df['Selling_Price']

# ==============================
# SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# CREATE MODEL
# ==============================

model = LinearRegression()

# ==============================
# TRAIN MODEL
# ==============================

model.fit(X_train, y_train)

# ==============================
# PREDICT VALUES
# ==============================

y_pred = model.predict(X_test)

# ==============================
# MODEL EVALUATION
# ==============================

score = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Model Accuracy (R2 Score): {score:.2f}")

print(f"Mean Absolute Error: {mae:.2f}")

# ==============================
# VISUALIZATION
# ==============================

plt.figure(figsize=(8,6))

# Scatter plot
plt.scatter(y_test, y_pred)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)

# Labels
plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Car Prices")

# Show graph
plt.show()