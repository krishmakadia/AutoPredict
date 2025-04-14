from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Create FastAPI instance
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing (can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data structure
class CarFeatures(BaseModel):
    Year: int
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: int      # 0 = CNG, 1 = Diesel, 2 = Petrol
    Seller_Type: int    # 0 = Dealer, 1 = Individual
    Transmission: int   # 0 = Automatic, 1 = Manual
    Owner: int

@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running"}

@app.post("/predict")
def predict_price(data: CarFeatures):
    input_data = np.array([[data.Year, data.Present_Price, data.Kms_Driven,
                            data.Fuel_Type, data.Seller_Type, data.Transmission, data.Owner]])

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(scaled_data)
    predicted_price = round(float(prediction[0]), 2)

    return {"predicted_price_lakhs": predicted_price}

