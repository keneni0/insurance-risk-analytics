import pytest
import pandas as pd
import numpy as np
import os


@pytest.fixture
def sample_df():
    """Create a sample insurance dataframe for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'PolicyID': range(1, 101),
        'TransactionMonth': pd.date_range('2024-01', periods=100, freq='D'),
        'Province': np.random.choice(['Ontario', 'Quebec', 'BC'], 100),
        'VehicleType': np.random.choice(['Sedan', 'SUV', 'Truck'], 100),
        'Gender': np.random.choice(['M', 'F'], 100),
        'Age': np.random.randint(18, 75, 100),
        'DrivingExperience': np.random.randint(0, 50, 100),
        'TotalPremium': np.random.uniform(500, 3000, 100),
        'TotalClaims': np.random.uniform(0, 2500, 100),
    })


class TestDVCIntegration:
    """Test suite for DVC data versioning integration."""
    
    def test_dvc_files_exist(self):
        """Test that DVC files are present in the project."""
        dvc_files = [
            'data/insurance_data_raw.csv.dvc',
            'data/insurance_data_cleaned.csv.dvc'
        ]
        for dvc_file in dvc_files:
            assert os.path.exists(dvc_file), f"Missing {dvc_file}"
    
    def test_dvc_config_exists(self):
        """Test that DVC configuration exists."""
        assert os.path.exists('.dvc/config'), "Missing .dvc/config"
    
    def test_gitignore_updated_for_dvc(self):
        """Test that .gitignore is configured for DVC."""
        with open('.gitignore', 'r') as f:
            content = f.read()
            assert 'data/*.csv' in content, ".gitignore not configured for DVC"
            assert '.dvc/cache' in content or '.dvc/tmp' in content


class TestDataPipeline:
    """Test suite for data pipeline operations."""
    
    def test_raw_data_exists(self):
        """Test that raw data file exists."""
        assert os.path.exists('data/insurance_data_raw.csv'), "Raw data file not found"
    
    def test_cleaned_data_exists(self):
        """Test that cleaned data file exists."""
        assert os.path.exists('data/insurance_data_cleaned.csv'), "Cleaned data file not found"
    
    def test_raw_data_has_records(self):
        """Test that raw data contains records."""
        df = pd.read_csv('data/insurance_data_raw.csv')
        assert len(df) > 0, "Raw data is empty"
    
    def test_cleaned_data_has_records(self):
        """Test that cleaned data contains records."""
        df = pd.read_csv('data/insurance_data_cleaned.csv')
        assert len(df) > 0, "Cleaned data is empty"
    
    def test_cleaned_data_is_subset(self):
        """Test that cleaned data is a subset of raw data."""
        raw = pd.read_csv('data/insurance_data_raw.csv')
        cleaned = pd.read_csv('data/insurance_data_cleaned.csv')
        assert len(cleaned) <= len(raw), "Cleaned data should be a subset of raw data"


class TestDataValidation:
    """Test suite for data validation in pipeline."""
    
    def test_no_negative_values_in_cleaned(self):
        """Test that cleaned data has no negative premium or claims."""
        df = pd.read_csv('data/insurance_data_cleaned.csv')
        assert (df['TotalPremium'] >= 0).all(), "Negative premium values found"
        assert (df['TotalClaims'] >= 0).all(), "Negative claim values found"
    
    def test_no_null_critical_values_in_cleaned(self):
        """Test that cleaned data has no null values in critical columns."""
        df = pd.read_csv('data/insurance_data_cleaned.csv')
        assert not df[['TotalPremium', 'TotalClaims']].isna().any().any()


class TestEDAMetrics:
    """Test suite for EDA metric calculations."""
    
    def test_loss_ratio_calculation(self, sample_df):
        """Test Loss Ratio metric calculation."""
        sample_df['LossRatio'] = sample_df['TotalClaims'] / sample_df['TotalPremium']
        assert (sample_df['LossRatio'] >= 0).all()
        assert (sample_df['LossRatio'] <= 1).any()  # Some should be reasonable
    
    def test_margin_calculation(self, sample_df):
        """Test Margin metric calculation."""
        sample_df['Margin'] = sample_df['TotalPremium'] - sample_df['TotalClaims']
        # Margin can be negative (losses), zero, or positive (profit)
        assert sample_df['Margin'].notna().all()
    
    def test_has_claim_indicator(self, sample_df):
        """Test HasClaim binary indicator."""
        sample_df['HasClaim'] = (sample_df['TotalClaims'] > 0).astype(int)
        assert sample_df['HasClaim'].isin([0, 1]).all()
        assert sample_df['HasClaim'].max() <= 1
    
    def test_segmentation_by_province(self, sample_df):
        """Test that data can be segmented by Province."""
        grouped = sample_df.groupby('Province')['TotalClaims'].mean()
        assert len(grouped) > 0
        assert grouped.notna().all()
    
    def test_temporal_aggregation(self, sample_df):
        """Test temporal aggregation by month."""
        sample_df['TransactionMonth'] = pd.to_datetime(sample_df['TransactionMonth'])
        monthly = sample_df.groupby(sample_df['TransactionMonth'].dt.to_period('M')).agg({
            'TotalClaims': 'mean',
            'TotalPremium': 'mean'
        })
        assert len(monthly) > 0


class TestFileStructure:
    """Test suite for project file structure."""
    
    def test_scripts_directory_exists(self):
        """Test that scripts directory exists."""
        assert os.path.isdir('scripts'), "Scripts directory not found"
    
    def test_generate_data_script_exists(self):
        """Test that data generation script exists."""
        assert os.path.exists('scripts/generate_data.py'), "generate_data.py not found"
    
    def test_clean_data_script_exists(self):
        """Test that data cleaning script exists."""
        assert os.path.exists('scripts/clean_data.py'), "clean_data.py not found"
    
    def test_notebooks_directory_exists(self):
        """Test that notebooks directory exists."""
        assert os.path.isdir('notebooks'), "Notebooks directory not found"
    
    def test_eda_notebook_exists(self):
        """Test that EDA notebook exists."""
        assert os.path.exists('notebooks/01_eda.ipynb'), "01_eda.ipynb not found"
    
    def test_tests_directory_exists(self):
        """Test that tests directory exists."""
        assert os.path.isdir('tests'), "Tests directory not found"
    
    def test_src_directory_exists(self):
        """Test that src directory exists."""
        assert os.path.isdir('src'), "src directory not found"
    
    def test_data_loader_exists(self):
        """Test that data_loader module exists."""
        assert os.path.exists('src/data_loader.py'), "data_loader.py not found"


class TestDocumentation:
    """Test suite for documentation."""
    
    def test_readme_exists(self):
        """Test that README file exists."""
        assert os.path.exists('README.md'), "README.md not found"
    
    def test_readme_has_content(self):
        """Test that README has substantial content."""
        with open('README.md', 'r') as f:
            content = f.read()
            assert len(content) > 500, "README is too short"
            assert 'DVC' in content, "README should mention DVC"
            assert 'Setup' in content or 'setup' in content, "README should have setup instructions"
    
    def test_requirements_txt_exists(self):
        """Test that requirements.txt exists."""
        assert os.path.exists('requirements.txt'), "requirements.txt not found"
    
    def test_requirements_has_dvc(self):
        """Test that DVC is in requirements."""
        with open('requirements.txt', 'r') as f:
            content = f.read()
            assert 'dvc' in content.lower(), "requirements.txt should include dvc"


class TestCICD:
    """Test suite for CI/CD pipeline configuration."""
    
    def test_github_actions_workflow_exists(self):
        """Test that GitHub Actions workflow file exists."""
        assert os.path.exists('.github/workflows/ci.yml'), "ci.yml workflow not found"
    
    def test_workflow_has_linting(self):
        """Test that workflow includes linting step."""
        with open('.github/workflows/ci.yml', 'r') as f:
            content = f.read()
            assert 'flake8' in content, "Workflow should include flake8 linting"
    
    def test_workflow_has_testing(self):
        """Test that workflow includes testing step."""
        with open('.github/workflows/ci.yml', 'r') as f:
            content = f.read()
            assert 'pytest' in content, "Workflow should include pytest"
