import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    return df
