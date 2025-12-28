import requests
import json

# Test single prediction
single_house = {
    "square_feet": 2000,
    "num_bedrooms": 3,
    "num_bathrooms": 2,
    "year_built": 2010,
    "location_quality": 7
}

response = requests.post(
    "http://localhost:8000/predict",
    json=single_house
)

print("Single Prediction:")
print(json.dumps(response.json(), indent=2))

# Test batch prediction
batch_houses = {
    "houses": [
        {
            "square_feet": 1500,
            "num_bedrooms": 2,
            "num_bathrooms": 1,
            "year_built": 1990,
            "location_quality": 5
        },
        {
            "square_feet": 3000,
            "num_bedrooms": 4,
            "num_bathrooms": 3,
            "year_built": 2020,
            "location_quality": 9
        }
    ]
}

response = requests.post(
    "http://localhost:8000/batch_predict",
    json=batch_houses
)

print("\nBatch Predictions:")
print(json.dumps(response.json(), indent=2))