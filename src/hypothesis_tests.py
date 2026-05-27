"""
Hypothesis testing module for risk analysis.

This module provides functions to test statistical hypotheses about insurance risk,
including claim frequency, claim severity, and margin differences across segments.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


def chi_squared_test(
    group_a: pd.Series,
    group_b: pd.Series,
    test_name: str = "Chi-Squared Test"
) -> Dict:
    """
    Perform chi-squared test on categorical data (binary outcomes).
    
    Used for testing claim frequency (binary: claim or no claim).
    
    Args:
        group_a: Binary series for control group (0/1)
        group_b: Binary series for test group (0/1)
        test_name: Name of the test for reporting
        
    Returns:
        Dict with chi_squared, p_value, effect_size
    """
    # Create contingency table
    contingency = pd.crosstab(
        pd.Series(['Control']*len(group_a) + ['Test']*len(group_b)),
        pd.Series(list(group_a) + list(group_b))
    )
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    # Calculate effect size (Cramér's V)
    n = len(group_a) + len(group_b)
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    result = {
        'test': test_name,
        'test_type': 'Chi-Squared',
        'statistic': chi2,
        'p_value': p_value,
        'effect_size': cramers_v,
        'group_a_mean': group_a.mean(),
        'group_b_mean': group_b.mean(),
        'difference': group_b.mean() - group_a.mean(),
        'significant': p_value < 0.05
    }
    
    logger.info(f"{test_name}: χ² = {chi2:.4f}, p = {p_value:.4f}")
    return result


def independent_t_test(
    group_a: pd.Series,
    group_b: pd.Series,
    test_name: str = "T-Test",
    equal_var: bool = False
) -> Dict:
    """
    Perform independent samples t-test for numerical data.
    
    Used for testing claim severity, margin, and other continuous metrics.
    
    Args:
        group_a: Numerical series for control group
        group_b: Numerical series for test group
        test_name: Name of the test for reporting
        equal_var: Whether to assume equal variance (Welch's t-test if False)
        
    Returns:
        Dict with t_statistic, p_value, effect_size (Cohen's d)
    """
    # Remove NaN values
    a_clean = group_a.dropna()
    b_clean = group_b.dropna()
    
    if len(a_clean) < 2 or len(b_clean) < 2:
        raise ValueError("Groups must have at least 2 non-NaN values")
    
    t_stat, p_value = stats.ttest_ind(a_clean, b_clean, equal_var=equal_var)
    
    # Calculate Cohen's d
    mean_a = a_clean.mean()
    mean_b = b_clean.mean()
    std_a = a_clean.std()
    std_b = b_clean.std()
    
    pooled_std = np.sqrt(((len(a_clean)-1)*std_a**2 + (len(b_clean)-1)*std_b**2) / 
                         (len(a_clean) + len(b_clean) - 2))
    cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
    
    result = {
        'test': test_name,
        'test_type': 'T-Test (Welch)' if not equal_var else 'T-Test',
        'statistic': t_stat,
        'p_value': p_value,
        'effect_size': cohens_d,
        'group_a_mean': mean_a,
        'group_b_mean': mean_b,
        'group_a_std': std_a,
        'group_b_std': std_b,
        'group_a_n': len(a_clean),
        'group_b_n': len(b_clean),
        'difference': mean_b - mean_a,
        'significant': p_value < 0.05
    }
    
    logger.info(f"{test_name}: t = {t_stat:.4f}, p = {p_value:.4f}, Cohen's d = {cohens_d:.4f}")
    return result


def mannwhitneyu_test(
    group_a: pd.Series,
    group_b: pd.Series,
    test_name: str = "Mann-Whitney U Test"
) -> Dict:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).
    
    Used when data is not normally distributed.
    
    Args:
        group_a: Series for control group
        group_b: Series for test group
        test_name: Name of the test for reporting
        
    Returns:
        Dict with U_statistic, p_value
    """
    a_clean = group_a.dropna()
    b_clean = group_b.dropna()
    
    u_stat, p_value = stats.mannwhitneyu(a_clean, b_clean, alternative='two-sided')
    
    result = {
        'test': test_name,
        'test_type': 'Mann-Whitney U',
        'statistic': u_stat,
        'p_value': p_value,
        'group_a_median': a_clean.median(),
        'group_b_median': b_clean.median(),
        'group_a_n': len(a_clean),
        'group_b_n': len(b_clean),
        'difference': b_clean.median() - a_clean.median(),
        'significant': p_value < 0.05
    }
    
    logger.info(f"{test_name}: U = {u_stat:.4f}, p = {p_value:.4f}")
    return result


def analyze_province_risk(df: pd.DataFrame) -> Dict:
    """
    H₀: There are no risk differences across provinces.
    
    Test claim severity differences between two provinces using t-test.
    (Claim frequency is 100% in dataset, so we test severity instead)
    """
    # Get two largest provinces for comparison
    province_counts = df['Province'].value_counts()
    prov_a, prov_b = province_counts.index[:2]
    
    # Filter for policies with claims
    df_claims = df[df['TotalClaims'] > 0]
    
    group_a = df_claims[df_claims['Province'] == prov_a]['TotalClaims']
    group_b = df_claims[df_claims['Province'] == prov_b]['TotalClaims']
    
    result = independent_t_test(group_a, group_b, test_name=f"Province Severity: {prov_a} vs {prov_b}")
    result['group_a_label'] = prov_a
    result['group_b_label'] = prov_b
    result['kpi'] = 'Claim Severity'
    
    return result


def analyze_zipcode_risk(df: pd.DataFrame) -> Dict:
    """
    H₀: There are no risk differences between zip codes.
    
    For this dataset, we'll use a proxy grouping and test Claim Severity.
    """
    # If data doesn't have zip code, create a synthetic grouping
    if 'PostalCode' not in df.columns:
        # Use province + vehicle type as proxy for geographic/demographic segments
        df['ZipGroup'] = df['Province'].str[:2]  # First 2 chars of province name
    else:
        df['ZipGroup'] = df['PostalCode'].astype(str).str[:3]
    
    zip_counts = df['ZipGroup'].value_counts()
    if len(zip_counts) < 2:
        return {
            'test': 'ZipCode Risk',
            'error': 'Insufficient zip code variation in dataset'
        }
    
    zip_a, zip_b = zip_counts.index[:2]
    
    # Filter for policies with claims
    df_claims = df[df['TotalClaims'] > 0]
    
    group_a = df_claims[df_claims['ZipGroup'] == zip_a]['TotalClaims']
    group_b = df_claims[df_claims['ZipGroup'] == zip_b]['TotalClaims']
    
    result = independent_t_test(group_a, group_b, test_name=f"ZipCode Severity: {zip_a} vs {zip_b}")
    result['group_a_label'] = str(zip_a)
    result['group_b_label'] = str(zip_b)
    result['kpi'] = 'Claim Severity'
    
    return result


def analyze_zipcode_margin(df: pd.DataFrame) -> Dict:
    """
    H₀: There is no significant margin (profit) difference between zip codes.
    
    Tests whether Margin (TotalPremium - TotalClaims) differs significantly by zip code.
    """
    if 'PostalCode' not in df.columns:
        df['ZipGroup'] = df['Province'].str[:2]
    else:
        df['ZipGroup'] = df['PostalCode'].astype(str).str[:3]
    
    zip_counts = df['ZipGroup'].value_counts()
    if len(zip_counts) < 2:
        return {
            'test': 'ZipCode Margin',
            'error': 'Insufficient zip code variation in dataset'
        }
    
    zip_a, zip_b = zip_counts.index[:2]
    
    group_a = df[df['ZipGroup'] == zip_a]['Margin']
    group_b = df[df['ZipGroup'] == zip_b]['Margin']
    
    result = independent_t_test(group_a, group_b, test_name=f"ZipCode Margin: {zip_a} vs {zip_b}")
    result['group_a_label'] = str(zip_a)
    result['group_b_label'] = str(zip_b)
    
    return result


def analyze_gender_risk(df: pd.DataFrame) -> Dict:
    """
    H₀: There is no significant risk difference between Women and Men.
    
    Tests claim severity differences between genders using t-test.
    """
    if 'Gender' not in df.columns or len(df['Gender'].unique()) < 2:
        return {
            'test': 'Gender Risk',
            'error': 'Gender column not available or insufficient variation'
        }
    
    genders = df['Gender'].unique()[:2]
    gender_a, gender_b = genders[0], genders[1]
    
    # Filter for policies with claims
    df_claims = df[df['TotalClaims'] > 0]
    
    group_a = df_claims[df_claims['Gender'] == gender_a]['TotalClaims']
    group_b = df_claims[df_claims['Gender'] == gender_b]['TotalClaims']
    
    result = independent_t_test(group_a, group_b, test_name=f"Gender Severity: {gender_a} vs {gender_b}")
    result['group_a_label'] = f"Gender {gender_a}"
    result['group_b_label'] = f"Gender {gender_b}"
    result['kpi'] = 'Claim Severity'
    
    return result


def format_hypothesis_result(result: Dict) -> str:
    """
    Format hypothesis test result as a readable string.
    
    Args:
        result: Dictionary from hypothesis test function
        
    Returns:
        Formatted result string
    """
    if 'error' in result:
        return f"❌ {result['test']}: {result['error']}"
    
    decision = "✅ REJECT H₀" if result['significant'] else "❌ FAIL TO REJECT H₀"
    
    p_val_str = f"{result['p_value']:.6f}"
    effect_str = f"{result.get('effect_size', 0):.4f}"
    
    return (
        f"{result['test']}\n"
        f"  Test Type: {result['test_type']}\n"
        f"  Statistic: {result['statistic']:.4f}\n"
        f"  P-value: {p_val_str}\n"
        f"  Effect Size: {effect_str}\n"
        f"  Group A Mean: {result.get('group_a_mean', result.get('group_a_median', 'N/A')):.4f}\n"
        f"  Group B Mean: {result.get('group_b_mean', result.get('group_b_median', 'N/A')):.4f}\n"
        f"  Decision: {decision}"
    )


def generate_business_interpretation(result: Dict) -> str:
    """
    Generate business-facing interpretation of hypothesis test result.
    
    Args:
        result: Dictionary from hypothesis test function
        
    Returns:
        Business interpretation string
    """
    if 'error' in result:
        return f"Unable to complete {result['test']}: {result['error']}"
    
    if not result['significant']:
        return f"No statistically significant difference found for {result['test']} (p = {result['p_value']:.4f}). Current segmentation strategy for this factor may be sufficient."
    
    diff_pct = (result['difference'] / result['group_a_mean'] * 100) if result['group_a_mean'] != 0 else 0
    
    group_a = result.get('group_a_label', 'Group A')
    group_b = result.get('group_b_label', 'Group B')
    
    interpretation = (
        f"We REJECT H₀ for {result['test'].lower()} (p < 0.05, p = {result['p_value']:.4f}).\n"
        f"{group_b} exhibits a {abs(diff_pct):.1f}% {'higher' if diff_pct > 0 else 'lower'} value "
        f"compared to {group_a}, suggesting a risk adjustment to our strategy may be warranted."
    )
    
    return interpretation
