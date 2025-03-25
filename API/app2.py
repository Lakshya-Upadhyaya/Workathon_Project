import pandas as pd
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model_path = "earthquake_prediction_model_0.pkl"
model = joblib.load(model_path)

# Load dataset for additional features
data_path = "balanced_processed_earthquake_data.csv"
df = pd.read_csv(data_path)

# Normalize input features
scaler = MinMaxScaler()
df[['latitude', 'longitude', 'depth']] = scaler.fit_transform(df[['latitude', 'longitude', 'depth']])

# Calculate mean values to fill missing data
feature_means = df.drop(columns=['status']).mean()

app = Flask(__name__)

# Route to render the map page
@app.route("/")
def home():
    return render_template("connect.html")  # Serve the frontend

# Route for earthquake probability prediction
@app.route("/predict", methods=["GET"])
def predict_earthquake_probability():
    lat = float(request.args.get("latitude"))
    lon = float(request.args.get("longitude"))
    depth = float(request.args.get("depth"))

    # Normalize input using the same scaler
    input_df = pd.DataFrame({"latitude": [lat], "longitude": [lon], "depth": [depth]})
    input_scaled = scaler.transform(input_df)  # This ensures correct feature ordering
    lat_norm, lon_norm, depth_norm = input_scaled[0]  # Extract normalized values



    # Find closest matching state in dataset
    df["distance"] = np.sqrt((df["latitude"] - lat_norm) ** 2 + (df["longitude"] - lon_norm) ** 2)
    closest_data = df.loc[df["distance"].idxmin()]

    # Prepare input for model
    input_data = pd.DataFrame({
        "latitude": [lat_norm],
        "longitude": [lon_norm],
        "depth": [depth_norm]
    })

    # Make prediction
    prob = model.predict_proba(input_data)[0][1] * 100  # Probability of earthquake

    return jsonify({"latitude": lat, "longitude": lon, "depth": depth, "probability": round(prob, 2)})

if __name__ == "__main__":
    app.run(debug=True)
