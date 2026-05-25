# Insurance Risk Analytics

A comprehensive data pipeline for exploratory data analysis (EDA) of insurance claims data with reproducible workflows using DVC (Data Version Control) and Git.

## Project Overview

This project demonstrates best practices for:
- **Data Versioning**: Using DVC to track different versions of datasets
- **Reproducible Workflows**: Ensuring analyses can be replicated with specific data versions
- **Code Organization**: Modular structure with reusable utilities
- **Quality Assurance**: Comprehensive testing with pytest
- **CI/CD Pipeline**: Automated linting and testing via GitHub Actions

## Project Structure

```
insurance-risk-analytics/
├── .dvc/                          # DVC configuration
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline
├── data/
│   ├── insurance_data_raw.csv.dvc       # Raw data tracking
│   ├── insurance_data_cleaned.csv.dvc   # Cleaned data tracking
│   └── .gitignore               # Excludes actual CSV files
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory Data Analysis
├── reports/                     # Output visualizations
├── scripts/
│   ├── generate_data.py         # Generate sample data
│   └── clean_data.py            # Data cleaning pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading with validation
│   └── eda_utils.py             # EDA utility functions
├── tests/                       # Pytest test suite
├── requirements.txt             # Python dependencies
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd insurance-risk-analytics
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC remote storage** (if not already configured)
   ```bash
   dvc remote add -d localstorage /path/to/dvc-storage
   ```

## Data Version Control (DVC)

### Overview

DVC enables reproducible data pipelines by versioning datasets separately from code. This is crucial for regulated industries where audit trails are required.

### Data Versions

The project tracks two data versions:

1. **`insurance_data_raw.csv`**: Raw, unfiltered dataset (5,000 records)
   - Source: Generated from `scripts/generate_data.py`
   - Contains all collected insurance claim records

2. **`insurance_data_cleaned.csv`**: Cleaned and filtered dataset (4,801 records)
   - Source: Processed via `scripts/clean_data.py`
   - Filters: Premium > $600, no duplicates, no missing values

### Reproducing the Data Pipeline

To get data locally from the remote storage:

```bash
# Fetch all tracked datasets
dvc pull

# Fetch specific version
dvc checkout
```

To update data and create a new version:

```bash
# Generate raw data
python scripts/generate_data.py

# Track the new version with DVC
dvc add data/insurance_data_raw.csv

# Push to remote storage
dvc push

# Commit .dvc file to Git
git add data/insurance_data_raw.csv.dvc
git commit -m "data: update raw data version"
```

### Viewing Data History

```bash
# Check what changed in tracked files
git log --oneline -- data/insurance_data_cleaned.csv.dvc

# Switch to a previous data version
git checkout <commit-hash> -- data/insurance_data_cleaned.csv.dvc
dvc checkout
```

## Running the EDA Notebook

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Open `notebooks/01_eda.ipynb`**

3. **Run cells in sequence** to:
   - Load data using the validated data loader
   - Explore data structure and quality
   - Create derived metrics (Loss Ratio, Margin, Claims Indicator)
   - Analyze patterns by Province, Vehicle Type, Gender
   - Visualize trends and relationships
   - Generate three required plots saved to `reports/`

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_data_loader.py -v
```

### Generate Coverage Report

```bash
pytest tests/ --cov=src --cov-report=html
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR:

1. **Linting**: `flake8` checks code style (max line length: 100)
2. **Testing**: `pytest` runs all tests
3. **Python Version**: 3.10

To run locally:
```bash
flake8 src/ --max-line-length=100
pytest tests/ -v
```

## Data Loader API

The `src/data_loader.py` module provides validated data loading:

```python
from src.data_loader import load_data

# Load with validation (default)
df = load_data('data/insurance_data_cleaned.csv')

# Load without validation
df = load_data('data/insurance_data_raw.csv', validate=False)
```

### Features

- ✅ Validates required columns
- ✅ Converts dates to datetime
- ✅ Detects and removes invalid data (negative values, missing critical fields)
- ✅ Comprehensive error messages
- ✅ Logging for audit trails

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes and test locally**
   ```bash
   pytest tests/ -v
   flake8 src/
   ```

3. **Commit with descriptive messages**
   ```bash
   git commit -m "feat: add new analysis"
   git commit -m "data: update cleaned dataset version"
   ```

4. **Push and create a Pull Request**
   ```bash
   git push origin feature/my-feature
   ```

## Key Commits in This Project

- **task-1**: Initialize project structure and CI pipeline
- **task-2**: Implement DVC data versioning with multiple data versions and comprehensive validation

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualization
- **scikit-learn**: Machine learning utilities
- **jupyter**: Interactive notebooks
- **dvc**: Data version control
- **pytest**: Testing framework
- **flake8**: Code linting
- **python-dotenv**: Environment variable management

## References

- [DVC Documentation](https://dvc.org/doc)
- [Git Best Practices](https://git-scm.com/book)
- [Pytest Documentation](https://docs.pytest.org)
- [Pandas API Reference](https://pandas.pydata.org/docs)