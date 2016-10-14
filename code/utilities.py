import pandas as pd

def read_data():
    """Read in the raw datasets and set the index"""

    train = pd.read_csv("../input/train.csv").set_index("id")
    test = pd.read_csv("../input/test.csv").set_index("id")

    return train, test

def column_check(df1, df2):
    """Remove any columns that don't exist in both datasets"""

    for column in df1.columns.values:
        if column not in df2.columns.values:
            df1 = df1.drop(column, axis=1)
    for column in df2.columns.values:
        if column not in df1.columns.values:
            df2 = df2.drop(column, axis=1)

    return df1, df2