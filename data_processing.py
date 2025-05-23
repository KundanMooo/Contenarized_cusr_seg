import pandas as pd

def load_and_preprocess(path: str) -> pd.DataFrame:
    """
    1) Read the raw TSV (tab-separated) file
    2) Compute Age, fill missing Income
    3) Return only the 4 columns we’ll cluster on
    """
    df = pd.read_csv(path, sep='\t')
    df['Age'] = 2022 - df['Year_Birth']
    df['Income'].fillna(df['Income'].median(), inplace=True)
    return df[['Age', 'Income', 'Recency', 'NumWebVisitsMonth']]
