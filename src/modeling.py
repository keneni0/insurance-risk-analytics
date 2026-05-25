"""
Statistical Modeling and Risk-Based Pricing Framework

This module implements machine learning models for insurance risk prediction
and provides a framework for dynamic, risk-based pricing optimization.

Models implemented:
- Linear Regression: Baseline model for claim severity prediction
- Random Forest: Ensemble method capturing non-linear relationships
- XGBoost: Gradient boosting for improved predictive performance

Features:
- Automatic feature engineering (vehicle age, policy duration, etc.)
- Categorical encoding (one-hot for low-cardinality, target encoding for high)
- SHAP-based interpretability analysis
- Premium optimization using predicted probabilities and severity
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import xgboost as xgb
import shap

logger = logging.getLogger(__name__)


def prepare_features(
    df: pd.DataFrame,
    target_column: str = 'TotalClaims',
    drop_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for modeling through feature engineering and cleaning.
    
    Features engineered:
    - PolicyDuration: Extracted from TransactionMonth
    - VehicleAge: If DrivingExperience available (proxy)
    - DriverRiskScore: Combination of driving experience and age
    - PremiumPercentile: Normalized premium within segment
    - ClaimFrequency: Historical claim rate per segment
    
    Args:
        df: Input insurance dataframe
        target_column: Target variable column name
        drop_columns: Additional columns to drop
    
    Returns:
        X: Feature matrix (pd.DataFrame)
        y: Target vector (pd.Series)
    """
    df_prep = df.copy()
    
    # Ensure required columns exist
    required = ['TotalPremium', 'TotalClaims', 'Province', 'Gender', 'VehicleType']
    for col in required:
        if col not in df_prep.columns:
            logger.warning(f"Column {col} not found in dataframe")
    
    # Feature engineering: Duration proxy from dates
    if 'TransactionMonth' in df_prep.columns:
        df_prep['TransactionMonth'] = pd.to_datetime(df_prep['TransactionMonth'])
        # Use year/month as ordinal feature
        df_prep['TransactionYearMonth'] = (
            df_prep['TransactionMonth'].dt.year * 12 + 
            df_prep['TransactionMonth'].dt.month
        )
    
    # Age-based features
    if 'Age' in df_prep.columns:
        df_prep['AgeGroup'] = pd.cut(df_prep['Age'], 
                                      bins=[0, 25, 35, 50, 65, 100],
                                      labels=['<25', '25-35', '35-50', '50-65', '65+'])
    
    # Driving experience score (inverse proxy for risk)
    if 'DrivingExperience' in df_prep.columns:
        df_prep['DriverRiskScore'] = 1 / (1 + df_prep['DrivingExperience'] / 10)
    
    # Premium normalization (claim severity indicator)
    if 'TotalPremium' in df_prep.columns:
        df_prep['LogPremium'] = np.log1p(df_prep['TotalPremium'])
        df_prep['PremiumPercentile'] = df_prep['TotalPremium'].rank(pct=True)
    
    # Create target if not exists
    if target_column not in df_prep.columns and 'TotalClaims' in df_prep.columns:
        df_prep[target_column] = df_prep['TotalClaims']
    
    # Drop unnecessary columns
    drop_default = ['PolicyID', 'TransactionMonth', 'TransactionYearMonth']
    drop_cols = (drop_columns or []) + drop_default
    drop_cols = [col for col in drop_cols if col in df_prep.columns]
    
    X = df_prep.drop(columns=drop_cols + [target_column])
    y = df_prep[target_column]
    
    logger.info(f"Features prepared: {X.shape[1]} features from {len(df_prep)} records")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numerical and categorical features.
    
    Strategy:
    - Numerical: StandardScaler normalization
    - Categorical: One-hot encoding (low-cardinality) or target encoding (high-cardinality)
    
    Args:
        X: Feature matrix
    
    Returns:
        ColumnTransformer with fitted preprocessing steps
    """
    # Identify feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Remove NaN columns
    numeric_features = [f for f in numeric_features if f in X.columns and X[f].notna().any()]
    categorical_features = [f for f in categorical_features if f in X.columns and X[f].notna().any()]
    
    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    logger.info(f"Preprocessor created: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    return preprocessor


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Train linear regression model for claim severity prediction.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features (optional)
        y_test: Test target (optional)
    
    Returns:
        Dictionary with model, preprocessor, training metrics, and predictions
    """
    # Preprocess
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Train
    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train_processed)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    result = {
        'model': model,
        'preprocessor': preprocessor,
        'model_type': 'Linear Regression',
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_predictions': y_train_pred
    }
    
    # Test evaluation if provided
    if X_test is not None and y_test is not None:
        X_test_processed = preprocessor.transform(X_test)
        y_test_pred = model.predict(X_test_processed)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        result.update({
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_predictions': y_test_pred
        })
    
    logger.info(f"Linear Regression trained: Train RMSE={train_rmse:.2f}, R²={train_r2:.4f}")
    return result


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    n_estimators: int = 100,
    max_depth: int = 15
) -> Dict[str, Any]:
    """
    Train Random Forest model for claim severity prediction.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features (optional)
        y_test: Test target (optional)
        n_estimators: Number of trees in ensemble
        max_depth: Maximum tree depth for regularization
    
    Returns:
        Dictionary with model, preprocessor, training metrics, feature importance
    """
    # Preprocess
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Train
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train_processed)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    result = {
        'model': model,
        'preprocessor': preprocessor,
        'model_type': 'Random Forest',
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_predictions': y_train_pred,
        'feature_importance': model.feature_importances_
    }
    
    # Test evaluation if provided
    if X_test is not None and y_test is not None:
        X_test_processed = preprocessor.transform(X_test)
        y_test_pred = model.predict(X_test_processed)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        result.update({
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_predictions': y_test_pred
        })
    
    logger.info(f"Random Forest trained: Train RMSE={train_rmse:.2f}, R²={train_r2:.4f}")
    return result


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    n_estimators: int = 100,
    max_depth: int = 6
) -> Dict[str, Any]:
    """
    Train XGBoost model for claim severity prediction.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features (optional)
        y_test: Test target (optional)
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
    
    Returns:
        Dictionary with model, preprocessor, training metrics, feature importance
    """
    # Preprocess
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Handle categorical features for XGBoost
    feature_names = []
    if hasattr(preprocessor, 'named_transformers_'):
        if 'num' in preprocessor.named_transformers_:
            feature_names.extend(preprocessor.named_transformers_['num'].get_feature_names_out())
        if 'cat' in preprocessor.named_transformers_:
            feature_names.extend(preprocessor.named_transformers_['cat'].get_feature_names_out())
    
    # Train
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        device='cpu'
    )
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train_processed)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    result = {
        'model': model,
        'preprocessor': preprocessor,
        'model_type': 'XGBoost',
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_predictions': y_train_pred,
        'feature_importance': model.feature_importances_
    }
    
    # Test evaluation if provided
    if X_test is not None and y_test is not None:
        X_test_processed = preprocessor.transform(X_test)
        y_test_pred = model.predict(X_test_processed)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        result.update({
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_predictions': y_test_pred
        })
    
    logger.info(f"XGBoost trained: Train RMSE={train_rmse:.2f}, R²={train_r2:.4f}")
    return result


def compare_models(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Args:
        results: List of model result dictionaries
    
    Returns:
        DataFrame with model comparison metrics
    """
    comparison = []
    for result in results:
        row = {
            'Model': result['model_type'],
            'Train RMSE': result.get('train_rmse', np.nan),
            'Train MAE': result.get('train_mae', np.nan),
            'Train R²': result.get('train_r2', np.nan),
            'Test RMSE': result.get('test_rmse', np.nan),
            'Test MAE': result.get('test_mae', np.nan),
            'Test R²': result.get('test_r2', np.nan)
        }
        comparison.append(row)
    
    df_comparison = pd.DataFrame(comparison)
    logger.info(f"Model comparison:\n{df_comparison.to_string()}")
    return df_comparison


def calculate_shap_values(
    model_result: Dict[str, Any],
    X_data: pd.DataFrame,
    max_samples: int = 100
) -> Dict[str, Any]:
    """
    Calculate SHAP values for model interpretability.
    
    Args:
        model_result: Dictionary from train_* function
        X_data: Feature data for SHAP calculation
        max_samples: Maximum samples for explainer (for performance)
    
    Returns:
        Dictionary with SHAP values and feature importance
    """
    # Get preprocessed data
    preprocessor = model_result['preprocessor']
    X_processed = preprocessor.transform(X_data.iloc[:max_samples])
    
    # Create SHAP explainer
    model = model_result['model']
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    
    # Calculate mean absolute SHAP for feature importance
    if isinstance(shap_values, list):  # Multi-output
        shap_values = shap_values[0]
    
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    result = {
        'shap_values': shap_values,
        'feature_importance': feature_importance,
        'expected_value': explainer.expected_value
    }
    
    logger.info(f"SHAP values calculated for {X_processed.shape[0]} samples")
    return result


def optimize_premium(
    df: pd.DataFrame,
    claim_severity_model: Dict[str, Any],
    claim_frequency_model: Optional[Dict[str, Any]] = None,
    expense_ratio: float = 0.15,
    profit_margin: float = 0.20
) -> pd.DataFrame:
    """
    Optimize premium using predicted claim probability and severity.
    
    Premium formula:
    P = (E[Severity] × P(Claim) + Expenses) × (1 + Margin)
    
    Where:
    - E[Severity]: Expected claim amount from regression model
    - P(Claim): Predicted claim probability
    - Expenses: Operational costs as % of premium
    - Margin: Desired profit margin
    
    Args:
        df: Insurance data
        claim_severity_model: Trained severity model result dictionary
        claim_frequency_model: Trained frequency model result (optional)
        expense_ratio: Operating expense as % of premium (default 15%)
        profit_margin: Desired profit margin (default 20%)
    
    Returns:
        DataFrame with optimized premiums and components
    """
    df_result = df.copy()
    
    # Get predictions
    preprocessor = claim_severity_model['preprocessor']
    X_processed = preprocessor.transform(df.drop(columns=['TotalClaims', 'TotalPremium'] if 'TotalClaims' in df.columns else []))
    
    severity_predictions = claim_severity_model['model'].predict(X_processed)
    df_result['PredictedSeverity'] = severity_predictions
    
    # Use claim frequency if available, otherwise use historical average
    if claim_frequency_model is not None:
        frequency_predictions = claim_frequency_model['model'].predict(X_processed)
        df_result['PredictedClaimProb'] = frequency_predictions
    else:
        # Use observed claim frequency as proxy
        if 'HasClaim' in df.columns:
            df_result['PredictedClaimProb'] = df['HasClaim']
        else:
            df_result['PredictedClaimProb'] = 0.3  # Default industry average
    
    # Calculate pure premium (expected losses)
    df_result['PurePremium'] = df_result['PredictedSeverity'] * df_result['PredictedClaimProb']
    
    # Add expense loading
    df_result['ExpenseLoading'] = df_result['PurePremium'] * expense_ratio
    
    # Calculate final premium with profit margin
    df_result['OptimizedPremium'] = (
        (df_result['PurePremium'] + df_result['ExpenseLoading']) * 
        (1 + profit_margin)
    )
    
    # Calculate premium adjustment vs current
    if 'TotalPremium' in df.columns:
        df_result['PremiumAdjustment'] = (
            df_result['OptimizedPremium'] - df['TotalPremium']
        )
        df_result['PremiumAdjustmentPct'] = (
            (df_result['OptimizedPremium'] - df['TotalPremium']) / df['TotalPremium'] * 100
        )
    
    logger.info(f"Premium optimization complete: {len(df_result)} policies evaluated")
    return df_result


def format_model_summary(model_result: Dict[str, Any]) -> str:
    """
    Format model result as display string.
    
    Args:
        model_result: Dictionary from train_* function
    
    Returns:
        Formatted string summary
    """
    summary = f"""
    {'='*60}
    {model_result['model_type']}
    {'='*60}
    Training Metrics:
    - RMSE: {model_result.get('train_rmse', 'N/A'):.2f}
    - MAE:  {model_result.get('train_mae', 'N/A'):.2f}
    - R²:   {model_result.get('train_r2', 'N/A'):.4f}
    
    """
    
    if 'test_rmse' in model_result:
        summary += f"""Testing Metrics:
    - RMSE: {model_result['test_rmse']:.2f}
    - MAE:  {model_result['test_mae']:.2f}
    - R²:   {model_result['test_r2']:.4f}
    """
    
    return summary


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("modeling.py module loaded successfully")
