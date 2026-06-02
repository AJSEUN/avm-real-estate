import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Locations and base price factors (Nigeria-specific logic)
locations = {
    "Ikoyi": 800000,
    "Victoria Island": 750000,
    "Lekki": 500000,
    "Surulere": 300000,
    "Yaba": 250000
}

property_types = ["Apartment", "Duplex"]

data = []

# Generate 300 rows
for _ in range(300):
    
    location = np.random.choice(list(locations.keys()))
    property_type = np.random.choice(property_types)
    
    bedrooms = np.random.randint(1, 6)
    bathrooms = bedrooms - np.random.randint(0, 2)
    size_sqft = np.random.randint(500, 4000)
    age = np.random.randint(0, 20)
    
    # Nigeria-specific price logic
    base_price = locations[location] * size_sqft
    
    # Adjustments
    bedroom_factor = bedrooms * 1000000
    bathroom_factor = bathrooms * 500000
    
    type_factor = 5000000 if property_type == "Duplex" else 0
    age_factor = -age * 200000
    
    # Final price
    price = base_price + bedroom_factor + bathroom_factor + type_factor + age_factor
    
    # Add noise for realism
    noise = np.random.randint(-2000000, 2000000)
    price += noise
    
    data.append([
        price,
        location,
        property_type,
        bedrooms,
        bathrooms,
        size_sqft,
        age
    ])

# Create DataFrame
columns = ["price", "location", "property_type", "bedrooms", "bathrooms", "size_sqft", "age"]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("synthetic_real_estate_data.csv", index=False)

print("✅ Synthetic dataset created successfully (300 rows)")
