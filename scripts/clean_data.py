import pandas as pd
import numpy as np

def clean_insurance_data(input_path: str, output_path: str):
    """
    Clean insurance data by removing duplicates, handling missing values,
    and ensuring data quality.
    """
    df = pd.read_csv(input_path)
    print(f"Loaded data: {df.shape[0]} records")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['PolicyID'])
    print(f"After removing duplicates: {df.shape[0]} records")
    
    # Handle missing values
    df = df.dropna(subset=['TotalPremium', 'TotalClaims'])
    print(f"After removing rows with missing premium/claims: {df.shape[0]} records")
    
    # Ensure non-negative values
    df = df[(df['TotalPremium'] >= 0) & (df['TotalClaims'] >= 0)]
    print(f"After removing negative values: {df.shape[0]} records")
    
    # Sort by transaction month
    df = df.sort_values('TransactionMonth').reset_index(drop=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    
    return df

if __name__ == '__main__':
    clean_insurance_data('data/insurance_data_raw.csv', 'data/insurance_data_cleaned.csv')
