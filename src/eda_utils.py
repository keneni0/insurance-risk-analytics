# EDA utility functions
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def calculate_loss_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate key loss metrics for insurance data.
    
    Args:
        df: Input dataframe with TotalPremium and TotalClaims columns
        
    Returns:
        pd.DataFrame: Input dataframe with added metric columns
    """
    df = df.copy()
    
    # Avoid division by zero
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    
    return df


def segment_analysis(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    """
    Perform segment analysis on insurance data.
    
    Args:
        df: Input dataframe with LossRatio column
        segment_column: Column to segment by
        
    Returns:
        pd.DataFrame: Summary statistics by segment
    """
    if 'LossRatio' not in df.columns:
        raise ValueError("DataFrame must contain 'LossRatio' column")
    
    summary = df.groupby(segment_column).agg({
        'LossRatio': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'TotalPremium': ['mean', 'sum'],
        'TotalClaims': ['mean', 'sum'],
    }).round(4)
    
    logger.info(f"Segment analysis complete for {segment_column}: {len(summary)} groups")
    return summary


def temporal_trend(df: pd.DataFrame, date_column: str = 'TransactionMonth') -> pd.DataFrame:
    """
    Calculate temporal trends in claims and frequency.
    
    Args:
        df: Input dataframe with date and HasClaim columns
        date_column: Name of date column
        
    Returns:
        pd.DataFrame: Time series of aggregated metrics
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in dataframe")
    
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    monthly = df.groupby(df[date_column].dt.to_period('M')).agg({
        'TotalClaims': ['mean', 'sum', 'count'],
        'HasClaim': 'mean',  # Claim frequency
        'TotalPremium': 'sum',
    }).round(4)
    
    monthly.columns = ['AvgClaims', 'TotalClaims', 'ClaimCount', 'ClaimFreq', 'TotalPremium']
    logger.info(f"Temporal analysis complete: {len(monthly)} periods")
    return monthly


def data_quality_report(df: pd.DataFrame) -> Dict:
    """
    Generate a data quality report for the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dict: Quality metrics and statistics
    """
    report = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isna().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
    }
    
    logger.info(f"Data quality report generated for {len(df)} records")
    return report


def identify_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> Tuple[int, pd.DataFrame]:
    """
    Identify outliers in a numeric column.
    
    Args:
        df: Input dataframe
        column: Column to analyze
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: Threshold for outlier detection (IQR multiplier or zscore limit)
        
    Returns:
        Tuple[int, pd.DataFrame]: Number of outliers and outlier rows
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > threshold]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Found {len(outliers)} outliers in {column} using {method}")
    return len(outliers), outliers

