from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import List, Dict, Any

# Import your DisasterAllocation class
from disaster_allocator import DisasterAllocation

app = FastAPI(title="Ujaagar Disaster Resource API")

# Configure CORS to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain instead of *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your data
@app.on_event("startup")
async def startup_event():
    global forecast_data, vehicle_data, model
    try:
        forecast_data = pd.read_csv("updated_forecast_dataset.csv")
        vehicle_data = pd.read_csv("updated_ngo_resource_centers.csv")
        # Load the trained model
        model = joblib.load("resource_predictor_model.pkl")
    except Exception as e:
        print(f"Error loading data or model: {e}")
        # We'll continue and handle errors in the endpoints

# Serve static files (your HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# API endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to Ujaagar Disaster Resource API"}

@app.get("/api/resource-centers")
async def get_resource_centers():
    """Get all resource centers with available resources"""
    try:
        result = vehicle_data.to_dict(orient="records")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving resource centers: {str(e)}")

@app.get("/api/districts")
async def get_districts():
    """Get all districts with population data"""
    try:
        result = forecast_data.to_dict(orient="records")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving districts: {str(e)}")

@app.get("/api/resource-demands")
async def get_resource_demands():
    """Calculate and return resource demands for all districts"""
    try:
        # Prepare the data for the model
        features = [
            "population", "male", "female", "literate", "workers",
            "male_workers", "female_workers", "literacy_rate", "worker_rate", "female_ratio"
        ]
        
        # Add calculated ratios if they don't exist yet
        if "literacy_rate" not in forecast_data.columns:
            forecast_data["literacy_rate"] = forecast_data["literate"] / forecast_data["population"]
            forecast_data["worker_rate"] = forecast_data["workers"] / forecast_data["population"]
            forecast_data["female_ratio"] = forecast_data["female"] / forecast_data["population"]
        
        # Get predictions from model
        predictions = model.predict(forecast_data[features])
        
        # Create result list
        result = []
        for i, district in enumerate(forecast_data.to_dict(orient="records")):
            # Find matching resource center
            resource_center = vehicle_data[vehicle_data["district_name"] == district["district_name"]].to_dict(orient="records")
            resource_center = resource_center[0] if resource_center else {}
            
            # Combine data
            district_result = {
                **district,
                "food_demand": int(predictions[i][0]),
                "medical_demand": int(predictions[i][1]),
                "shelter_demand": int(predictions[i][2]),
                "food_available": resource_center.get("food_available", 0),
                "medical_available": resource_center.get("medical_available", 0),
                "shelter_available": resource_center.get("shelter_available", 0),
                "camp_name": resource_center.get("camp_name", "No camp in this district")
            }
            result.append(district_result)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating resource demands: {str(e)}")

@app.get("/api/allocate-resources/{district_name}")
async def allocate_resources(district_name: str):
    """Allocate resources for a specific district"""
    try:
        # Find the district
        district = forecast_data[forecast_data["district_name"] == district_name]
        if district.empty:
            raise HTTPException(status_code=404, detail=f"District {district_name} not found")
        
        # Get the demands
        features = [
            "population", "male", "female", "literate", "workers",
            "male_workers", "female_workers", "literacy_rate", "worker_rate", "female_ratio"
        ]
        
        # Calculate ratios if needed
        if "literacy_rate" not in district.columns:
            district["literacy_rate"] = district["literate"] / district["population"]
            district["worker_rate"] = district["workers"] / district["population"]
            district["female_ratio"] = district["female"] / district["population"]
        
        prediction = model.predict(district[features])
        
        demands = {
            "food_demand": int(prediction[0][0]),
            "medical_demand": int(prediction[0][1]),
            "shelter_demand": int(prediction[0][2])
        }
        
        # Use your allocation system
        allocation_system = DisasterAllocation(None, forecast_data, vehicle_data)
        allocations, unmet_demands = allocation_system.allocate_resources(demands)
        
        return {
            "district_name": district_name,
            "demands": demands,
            "allocations": allocations,
            "unmet_demands": unmet_demands
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error allocating resources: {str(e)}")

# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)