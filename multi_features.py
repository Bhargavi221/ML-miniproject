"""
Generate lagged features for machine learning
Author: Srush
"""

import pandas as pd

def create_lagged_features(df, lags=5):
    """
    Generate lagged features for each column in the DataFrame.
    Returns a DataFrame with lagged columns.
    """
    lagged = pd.DataFrame(index=df.index)

    for col in df.columns:
        for i in range(1, lags + 1):
            lagged[f"{col}_lag{i}"] = df[col].shift(i)

    lagged = lagged.dropna()
    return lagged
