import pytest
import pandas as pd
import numpy as np
from src.eda_utils import (
    calculate_loss_metrics,
    segment_analysis,
    temporal_trend,
    data_quality_report,
    identify_outliers
)


@pytest.fixture
def sample_df():
    """Create sample insurance data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'PolicyID': range(1, 101),
        'TransactionMonth': pd.date_range('2024-01', periods=100, freq='D'),
        'Province': np.random.choice(['Ontario', 'Quebec', 'BC'], 100),
        'TotalPremium': np.random.uniform(500, 3000, 100),
        'TotalClaims': np.random.uniform(0, 2500, 100),
    })


class TestLossMetrics:
    """Test loss metric calculations."""
    
    def test_calculate_loss_metrics_adds_columns(self, sample_df):
        """Test that calculate_loss_metrics adds required columns."""
        result = calculate_loss_metrics(sample_df)
        assert 'LossRatio' in result.columns
        assert 'Margin' in result.columns
        assert 'HasClaim' in result.columns
    
    def test_loss_ratio_calculation(self, sample_df):
        """Test LossRatio calculation."""
        result = calculate_loss_metrics(sample_df)
        expected = (sample_df['TotalClaims'] / sample_df['TotalPremium']).rename('LossRatio')
        pd.testing.assert_series_equal(result['LossRatio'], expected, check_names=True)
    
    def test_margin_calculation(self, sample_df):
        """Test Margin calculation."""
        result = calculate_loss_metrics(sample_df)
        expected = (sample_df['TotalPremium'] - sample_df['TotalClaims']).rename('Margin')
        pd.testing.assert_series_equal(result['Margin'], expected, check_names=True)
    
    def test_has_claim_indicator(self, sample_df):
        """Test HasClaim binary indicator."""
        result = calculate_loss_metrics(sample_df)
        assert result['HasClaim'].isin([0, 1]).all()


class TestSegmentAnalysis:
    """Test segment analysis functionality."""
    
    def test_segment_analysis_by_province(self, sample_df):
        """Test segmentation by province."""
        df = calculate_loss_metrics(sample_df)
        result = segment_analysis(df, 'Province')
        assert len(result) > 0
        assert all(prov in result.index for prov in df['Province'].unique())
    
    def test_segment_analysis_missing_loss_ratio(self, sample_df):
        """Test that segment_analysis raises error if LossRatio missing."""
        with pytest.raises(ValueError, match="LossRatio"):
            segment_analysis(sample_df, 'Province')
    
    def test_segment_analysis_returns_dataframe(self, sample_df):
        """Test that segment_analysis returns a dataframe."""
        df = calculate_loss_metrics(sample_df)
        result = segment_analysis(df, 'Province')
        assert isinstance(result, pd.DataFrame)


class TestTemporalTrend:
    """Test temporal trend analysis."""
    
    def test_temporal_trend_returns_dataframe(self, sample_df):
        """Test that temporal_trend returns a dataframe."""
        df = calculate_loss_metrics(sample_df)
        result = temporal_trend(df)
        assert isinstance(result, pd.DataFrame)
    
    def test_temporal_trend_has_required_columns(self, sample_df):
        """Test that temporal trend has expected columns."""
        df = calculate_loss_metrics(sample_df)
        result = temporal_trend(df)
        expected_cols = ['AvgClaims', 'TotalClaims', 'ClaimCount', 'ClaimFreq', 'TotalPremium']
        assert all(col in result.columns for col in expected_cols)
    
    def test_temporal_trend_missing_date_column(self, sample_df):
        """Test error when date column is missing."""
        df = sample_df.drop('TransactionMonth', axis=1)
        with pytest.raises(ValueError, match="not found"):
            temporal_trend(df)
    
    def test_temporal_trend_claim_freq_in_range(self, sample_df):
        """Test that claim frequency is between 0 and 1."""
        df = calculate_loss_metrics(sample_df)
        result = temporal_trend(df)
        assert (result['ClaimFreq'] >= 0).all()
        assert (result['ClaimFreq'] <= 1).all()


class TestDataQualityReport:
    """Test data quality reporting."""
    
    def test_data_quality_report_returns_dict(self, sample_df):
        """Test that report returns a dictionary."""
        report = data_quality_report(sample_df)
        assert isinstance(report, dict)
    
    def test_data_quality_report_has_keys(self, sample_df):
        """Test that report contains expected keys."""
        report = data_quality_report(sample_df)
        expected_keys = ['total_records', 'total_columns', 'missing_values', 'duplicate_rows', 'data_types']
        assert all(key in report for key in expected_keys)
    
    def test_data_quality_report_values(self, sample_df):
        """Test that report values are correct."""
        report = data_quality_report(sample_df)
        assert report['total_records'] == len(sample_df)
        assert report['total_columns'] == len(sample_df.columns)
        assert report['duplicate_rows'] == 0


class TestOutlierDetection:
    """Test outlier detection."""
    
    def test_outlier_detection_iqr(self, sample_df):
        """Test IQR method for outlier detection."""
        count, outliers = identify_outliers(sample_df, 'TotalPremium', method='iqr')
        assert isinstance(count, int)
        assert isinstance(outliers, pd.DataFrame)
        assert len(outliers) == count
    
    def test_outlier_detection_zscore(self, sample_df):
        """Test z-score method for outlier detection."""
        count, outliers = identify_outliers(sample_df, 'TotalClaims', method='zscore', threshold=3)
        assert isinstance(count, int)
        assert len(outliers) == count
    
    def test_outlier_detection_invalid_column(self, sample_df):
        """Test error with invalid column."""
        with pytest.raises(ValueError, match="not found"):
            identify_outliers(sample_df, 'NonexistentColumn')
    
    def test_outlier_detection_invalid_method(self, sample_df):
        """Test error with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            identify_outliers(sample_df, 'TotalPremium', method='invalid')
