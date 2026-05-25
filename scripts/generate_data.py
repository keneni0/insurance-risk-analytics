import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_insurance_data(num_records=5000, seed=42):
    """Generate sample insurance claims data for testing and demo purposes."""
    np.random.seed(seed)
    
    # Date range
    end_date = datetime(2024, 12, 31)
    start_date = datetime(2020, 1, 1)
    date_range = end_date - start_date
    
    # Generate random transactions
    data = {
        'PolicyID': range(1, num_records + 1),
        'TransactionMonth': [start_date + timedelta(days=int(x)) for x in 
                             np.random.randint(0, date_range.days, num_records)],
        'Province': np.random.choice(['Ontario', 'Quebec', 'British Columbia', 'Alberta', 'Manitoba'], num_records),
        'VehicleType': np.random.choice(['Sedan', 'SUV', 'Truck', 'Van'], num_records),
        'Gender': np.random.choice(['M', 'F'], num_records),
        'Age': np.random.randint(18, 75, num_records),
        'DrivingExperience': np.random.randint(0, 50, num_records),
        'TotalPremium': np.random.uniform(500, 3000, num_records),
        'TotalClaims': np.random.uniform(0, 2500, num_records),
    }
    
    df = pd.DataFrame(data)
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth']).dt.to_period('M').astype(str)
    df = df.sort_values('TransactionMonth').reset_index(drop=True)
    
    return df

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate and save raw data
    df = generate_insurance_data(num_records=5000)
    df.to_csv('data/insurance_data_raw.csv', index=False)
    print(f"Generated raw data: {df.shape[0]} records saved to data/insurance_data_raw.csv")
    print(df.head())
