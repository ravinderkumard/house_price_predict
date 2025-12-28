import pandas as pd
import numpy as np

# Create synthetic house price data
np.random.seed(42)
n_samples = 1000

data = {
    'square_feet': np.random.randint(800, 4000, n_samples),
    'num_bedrooms': np.random.randint(1, 6, n_samples),
    'num_bathrooms': np.random.randint(1, 5, n_samples),
    'year_built': np.random.randint(1950, 2023, n_samples),
    'location_quality': np.random.randint(1, 11, n_samples),  # 1-10 scale
}

print("Process complete")

base_price = 50000 # bias

price = (
    base_price 
    +data['square_feet']*100
    +data['num_bedrooms']*20000
    +data['num_bathrooms']*15000
    +(data['year_built']-1950)* 500
    +data['location_quality']*10000
    +np.random.normal(0,20000,n_samples)
)

data['price'] = price.astype(int)

df = pd.DataFrame(data)
df.to_csv("data/house_data.csv",index=False)

print(f"Created dataset with {len(df)} samples")
print(df.head)