import pandas as pd
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path: str, validate: bool = True) -> pd.DataFrame:
    """
    Load insurance data from CSV file with validation and error handling.
    
    Args:
        path: Path to the CSV file
        validate: Whether to validate data quality
        
    Returns:
        pd.DataFrame: Loaded and processed dataframe
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails or required columns are missing
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    try:
        df = pd.read_csv(path, low_memory=False)
        logger.info(f"Loaded {len(df)} records from {path}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    # Validate required columns
    required_cols = ['PolicyID', 'TransactionMonth', 'TotalPremium', 'TotalClaims']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert TransactionMonth to datetime
    try:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    except Exception as e:
        raise ValueError(f"Failed to parse TransactionMonth: {e}")
    
    if validate:
        df = validate_data(df)
    
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean insurance data.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Validated dataframe
        
    Raises:
        ValueError: If data validation fails
    """
    # Check for negative values in numeric columns
    numeric_cols = ['TotalPremium', 'TotalClaims']
    for col in numeric_cols:
        if (df[col] < 0).any():
            logger.warning(f"Found negative values in {col}, removing them")
            df = df[df[col] >= 0]
    
    # Check for missing critical values
    critical_cols = ['TotalPremium', 'TotalClaims']
    if df[critical_cols].isna().any().any():
        logger.warning(f"Found missing values in critical columns, removing rows")
        df = df.dropna(subset=critical_cols)
    
    logger.info(f"Validation complete: {len(df)} records")
    return df
