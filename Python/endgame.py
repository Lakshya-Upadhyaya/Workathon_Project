

import pandas as pd

data = pd.read_csv("all_month.csv")
data

data.columns

data.isnull().sum()

columns_to_drop = ['id', 'updated', 'place', 'locationSource', 'magSource','mag','magType','nst','gap','dmin','rms','net','type','horizontalError','depthError','magError','magError','magnst','locationSource','magSource']
data = data.drop(columns=columns_to_drop, errors='ignore')

columns_to_drop = ['magNst','time']
data = data.drop(columns=columns_to_drop, errors='ignore')



data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical_features = ['latitude', 'longitude', 'depth']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

data

data.to_csv("processed_earthquake_data.csv", index=False)

processed_earthquake_data = pd.read_csv("processed_earthquake_data.csv")
processed_earthquake_data

from sklearn.cluster import KMeans
from scipy.stats import zscore

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

kmeans = KMeans(n_clusters=5, random_state=42)
processed_earthquake_data['region_cluster'] = kmeans.fit_predict(processed_earthquake_data[['latitude', 'longitude']])

z_scores = np.abs(zscore(processed_earthquake_data[['depth']]))
processed_earthquake_data = processed_earthquake_data[(z_scores < 3).all(axis=1)]

processed_earthquake_data

processed_earthquake_data['status'].value_counts()

from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()
processed_earthquake_data['status'] = label_encoder.fit_transform(processed_earthquake_data['status'])


print(processed_earthquake_data['status'].value_counts())  # Check encoding

from imblearn.over_sampling import SMOTE

# Define features (X) and target (y)
X = processed_earthquake_data.drop(columns=['status'])  # Features
y = processed_earthquake_data['status']  # Target variable

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back to DataFrame
balanced_earthquake_data = pd.DataFrame(X_resampled, columns=X.columns)
balanced_earthquake_data['status'] = y_resampled  # Add target back

# Check class balance
print(balanced_earthquake_data['status'].value_counts())

balanced_earthquake_data

plt.figure(figsize=(25, 12))
sns.heatmap(processed_earthquake_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(x=processed_earthquake_data['longitude'], y=processed_earthquake_data['latitude'],  palette='viridis', alpha=0.7)
plt.title("Earthquake Distribution by Region Clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

balanced_earthquake_data.to_csv("balanced_processed_earthquake_data.csv", index=False)

df = pd.read_csv('balanced_processed_earthquake_data.csv')
df

df.columns

df.isnull().sum()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import zscore

df['status'].value_counts()

df

X = df.drop(columns=['status'])  # Features
y = df['status']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

import joblib
import os

# Define the filename
model_filename = "earthquake_prediction_model_0.pkl"

# Check if the model has the 'predict' and 'predict_proba' attributes
if hasattr(model, "predict") and hasattr(model, "predict_proba"):
    # Save the trained model
    joblib.dump(model, model_filename)
    print(f"Model saved successfully as {model_filename}")

    # Verify that the model was saved correctly
    if os.path.exists(model_filename):
        print("File exists and is ready to be loaded.")
    else:
        print("Warning: File not found after saving!")
else:
    print("Error: The object being saved is not a valid classifier model.")

import joblib

# Load the saved model
model_filename = "earthquake_prediction_model_0.pkl"
model = joblib.load(model_filename)

# Check the type of the model
print(type(model))

