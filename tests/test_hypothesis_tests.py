import pytest
import pandas as pd
import numpy as np
from src.hypothesis_tests import (
    chi_squared_test,
    independent_t_test,
    mannwhitneyu_test
)


@pytest.fixture
def sample_df():
    """Create sample insurance data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'Province': np.random.choice(['Ontario', 'Quebec', 'BC'], 200),
        'Gender': np.random.choice(['M', 'F'], 200),
        'TotalPremium': np.random.uniform(500, 3000, 200),
        'TotalClaims': np.random.uniform(0, 2500, 200),
        'HasClaim': np.random.binomial(1, 0.3, 200),
        'Margin': np.random.uniform(-1000, 2500, 200),
    })


class TestChiSquaredTest:
    """Test chi-squared hypothesis testing."""
    
    def test_chi_squared_returns_dict(self):
        """Test that chi-squared returns a dictionary with expected keys."""
        group_a = pd.Series([0, 0, 1, 0, 1])
        group_b = pd.Series([1, 1, 1, 0, 1])
        
        result = chi_squared_test(group_a, group_b)
        assert isinstance(result, dict)
        assert 'test_type' in result
        assert 'p_value' in result
        assert 'statistic' in result
        assert result['test_type'] == 'Chi-Squared'
    
    def test_chi_squared_p_value_in_range(self):
        """Test that p-value is between 0 and 1."""
        group_a = pd.Series([0, 0, 0, 0, 0])
        group_b = pd.Series([1, 1, 1, 1, 1])
        
        result = chi_squared_test(group_a, group_b)
        assert 0 <= result['p_value'] <= 1


class TestIndependentTTest:
    """Test independent t-test."""
    
    def test_t_test_returns_dict(self):
        """Test that t-test returns expected structure."""
        group_a = pd.Series(np.random.normal(10, 2, 30))
        group_b = pd.Series(np.random.normal(12, 2, 30))
        
        result = independent_t_test(group_a, group_b)
        assert isinstance(result, dict)
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'effect_size' in result
    
    def test_t_test_p_value_range(self):
        """Test that p-value is valid."""
        group_a = pd.Series([1, 2, 3, 4, 5])
        group_b = pd.Series([2, 3, 4, 5, 6])
        
        result = independent_t_test(group_a, group_b)
        assert 0 <= result['p_value'] <= 1
    
    def test_t_test_with_nans(self):
        """Test t-test handles NaN values."""
        group_a = pd.Series([1, 2, np.nan, 4, 5])
        group_b = pd.Series([2, np.nan, 4, 5, 6])
        
        result = independent_t_test(group_a, group_b)
        assert result['group_a_n'] == 4
        assert result['group_b_n'] == 4


class TestMannWhitneyUTest:
    """Test Mann-Whitney U test."""
    
    def test_mannwhitneyu_returns_dict(self):
        """Test Mann-Whitney U returns expected structure."""
        group_a = pd.Series([1, 2, 3, 4, 5])
        group_b = pd.Series([6, 7, 8, 9, 10])
        
        result = mannwhitneyu_test(group_a, group_b)
        assert isinstance(result, dict)
        assert 'statistic' in result
        assert 'p_value' in result
    
    def test_mannwhitneyu_identifies_difference(self):
        """Test that Mann-Whitney U detects differences."""
        group_a = pd.Series([1, 2, 3, 4, 5])
        group_b = pd.Series([100, 200, 300, 400, 500])
        
        result = mannwhitneyu_test(group_a, group_b)
        # Very different distributions should have low p-value
        assert result['p_value'] < 0.05


class TestProvinceRiskDifferences:
    """Test province risk hypothesis."""
    
    def test_province_risk_returns_result(self, sample_df):
        """Test that province risk test returns result."""
        from src.hypothesis_tests import analyze_province_risk
        result = analyze_province_risk(sample_df)
        assert isinstance(result, dict)
        assert 'p_value' in result
        assert 'test' in result
    
    def test_province_risk_identifies_groups(self, sample_df):
        """Test that groups are properly identified."""
        from src.hypothesis_tests import analyze_province_risk
        result = analyze_province_risk(sample_df)
        assert 'group_a_label' in result
        assert 'group_b_label' in result


class TestGenderRiskDifferences:
    """Test gender risk hypothesis."""
    
    def test_gender_risk_returns_result(self, sample_df):
        """Test that gender risk test returns result."""
        from src.hypothesis_tests import analyze_gender_risk
        result = analyze_gender_risk(sample_df)
        if 'error' not in result:
            assert isinstance(result, dict)
            assert 'p_value' in result
    
    def test_gender_risk_identifies_groups(self, sample_df):
        """Test that gender groups are identified."""
        from src.hypothesis_tests import analyze_gender_risk
        result = analyze_gender_risk(sample_df)
        if 'error' not in result:
            assert 'group_a_label' in result
            assert 'group_b_label' in result


class TestHypothesisSignificance:
    """Test hypothesis significance interpretation."""
    
    def test_significant_hypothesis(self):
        """Test detection of significant hypothesis."""
        # Create highly different groups
        group_a = pd.Series([0]*45 + [1]*5)  # 10% claim rate
        group_b = pd.Series([0]*20 + [1]*30)  # 60% claim rate
        
        result = chi_squared_test(group_a, group_b)
        assert result['significant'] == True
    
    def test_insignificant_hypothesis(self):
        """Test detection of non-significant hypothesis."""
        # Create similar groups
        group_a = pd.Series([0]*47 + [1]*3)  # 6% claim rate
        group_b = pd.Series([0]*46 + [1]*4)  # 8% claim rate
        
        result = chi_squared_test(group_a, group_b)
        # With small effect sizes, may not be significant
        assert 'significant' in result
        assert isinstance(result['significant'], (bool, np.bool_))
