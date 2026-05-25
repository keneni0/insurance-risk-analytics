import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.data_loader import load_data, validate_data


@pytest.fixture
def valid_csv_file():
    """Create a valid test CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("PolicyID,TransactionMonth,Province,VehicleType,Gender,Age,DrivingExperience,TotalPremium,TotalClaims\n")
        f.write("1,2024-01,Ontario,Sedan,M,35,10,1500,500\n")
        f.write("2,2024-02,Quebec,SUV,F,42,15,2000,800\n")
        f.write("3,2024-03,BC,Truck,M,28,5,1200,300\n")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def invalid_csv_missing_columns():
    """Create a CSV file with missing required columns."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("PolicyID,Province,VehicleType\n")
        f.write("1,Ontario,Sedan\n")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def csv_with_negative_values():
    """Create a CSV file with negative premium/claims values."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("PolicyID,TransactionMonth,Province,VehicleType,Gender,Age,DrivingExperience,TotalPremium,TotalClaims\n")
        f.write("1,2024-01,Ontario,Sedan,M,35,10,-1500,500\n")
        f.write("2,2024-02,Quebec,SUV,F,42,15,2000,-800\n")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def csv_with_missing_values():
    """Create a CSV file with missing critical values."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("PolicyID,TransactionMonth,Province,VehicleType,Gender,Age,DrivingExperience,TotalPremium,TotalClaims\n")
        f.write("1,2024-01,Ontario,Sedan,M,35,10,,500\n")
        f.write("2,2024-02,Quebec,SUV,F,42,15,2000,\n")
        f.write("3,2024-03,BC,Truck,M,28,5,1200,300\n")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


class TestLoadData:
    """Test suite for load_data function."""
    
    def test_load_valid_data(self, valid_csv_file):
        """Test loading valid CSV file."""
        df = load_data(valid_csv_file)
        assert len(df) == 3
        assert 'TransactionMonth' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['TransactionMonth'])
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_data('/nonexistent/path/file.csv')
    
    def test_load_missing_required_columns(self, invalid_csv_missing_columns):
        """Test loading CSV with missing required columns."""
        with pytest.raises(ValueError, match="Missing required columns"):
            load_data(invalid_csv_missing_columns)
    
    def test_load_with_validation_removes_negative(self, csv_with_negative_values):
        """Test that validation removes rows with negative values."""
        df = load_data(csv_with_negative_values, validate=True)
        assert len(df) == 0  # All rows have negative values
    
    def test_load_without_validation(self, csv_with_negative_values):
        """Test loading without validation keeps all rows."""
        df = load_data(csv_with_negative_values, validate=False)
        assert len(df) == 2  # All rows loaded
    
    def test_load_removes_missing_values(self, csv_with_missing_values):
        """Test that validation removes rows with missing critical values."""
        df = load_data(csv_with_missing_values, validate=True)
        assert len(df) == 1  # Only row 3 has complete data


class TestValidateData:
    """Test suite for validate_data function."""
    
    def test_validate_removes_negative_premium(self):
        """Test that validation removes rows with negative premium."""
        df = pd.DataFrame({
            'TotalPremium': [1000, -500, 2000],
            'TotalClaims': [300, 200, 500]
        })
        result = validate_data(df)
        assert len(result) == 2
        assert (result['TotalPremium'] >= 0).all()
    
    def test_validate_removes_negative_claims(self):
        """Test that validation removes rows with negative claims."""
        df = pd.DataFrame({
            'TotalPremium': [1000, 500, 2000],
            'TotalClaims': [300, -200, 500]
        })
        result = validate_data(df)
        assert len(result) == 2
        assert (result['TotalClaims'] >= 0).all()
    
    def test_validate_removes_missing_values(self):
        """Test that validation removes rows with missing critical values."""
        df = pd.DataFrame({
            'TotalPremium': [1000, np.nan, 2000],
            'TotalClaims': [300, 200, 500]
        })
        result = validate_data(df)
        assert len(result) == 2
    
    def test_validate_preserves_valid_data(self):
        """Test that validation preserves all valid data."""
        df = pd.DataFrame({
            'TotalPremium': [1000, 500, 2000],
            'TotalClaims': [300, 200, 500]
        })
        result = validate_data(df)
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, df)


class TestDataQuality:
    """Test suite for data quality checks."""
    
    def test_transaction_month_datetime_type(self, valid_csv_file):
        """Test that TransactionMonth is converted to datetime."""
        df = load_data(valid_csv_file)
        assert pd.api.types.is_datetime64_any_dtype(df['TransactionMonth'])
    
    def test_no_duplicates_in_policy_id(self, valid_csv_file):
        """Test that there are no duplicate Policy IDs (when valid)."""
        df = load_data(valid_csv_file)
        assert df['PolicyID'].is_unique
    
    def test_required_columns_present(self, valid_csv_file):
        """Test that all required columns are present."""
        df = load_data(valid_csv_file)
        required_cols = ['PolicyID', 'TransactionMonth', 'TotalPremium', 'TotalClaims']
        assert all(col in df.columns for col in required_cols)


class TestEdgeCases:
    """Test suite for edge cases."""
    
    def test_load_single_row(self):
        """Test loading data with a single row."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("PolicyID,TransactionMonth,Province,VehicleType,Gender,Age,DrivingExperience,TotalPremium,TotalClaims\n")
            f.write("1,2024-01,Ontario,Sedan,M,35,10,1500,500\n")
            temp_path = f.name
        
        try:
            df = load_data(temp_path)
            assert len(df) == 1
        finally:
            os.unlink(temp_path)
    
    def test_load_zero_claims(self):
        """Test loading data where TotalClaims is zero."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("PolicyID,TransactionMonth,Province,VehicleType,Gender,Age,DrivingExperience,TotalPremium,TotalClaims\n")
            f.write("1,2024-01,Ontario,Sedan,M,35,10,1500,0\n")
            temp_path = f.name
        
        try:
            df = load_data(temp_path)
            assert len(df) == 1
            assert df.iloc[0]['TotalClaims'] == 0
        finally:
            os.unlink(temp_path)
    
    def test_large_values(self):
        """Test loading data with very large premium/claims values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("PolicyID,TransactionMonth,Province,VehicleType,Gender,Age,DrivingExperience,TotalPremium,TotalClaims\n")
            f.write("1,2024-01,Ontario,Sedan,M,35,10,999999,999999\n")
            temp_path = f.name
        
        try:
            df = load_data(temp_path)
            assert len(df) == 1
            assert df.iloc[0]['TotalPremium'] == 999999
        finally:
            os.unlink(temp_path)
