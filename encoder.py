import pandas as pd

def ordinal_encoder(col: pd.Series, mapping: dict | None = None) -> pd.Series:
    """
    Encode categorical variable as ordinal.

    Parameters:
    -----------
    col : pd.Series
        The categorical variable to be encoded.

    Returns:
    --------
    pd.Series
        The encoded categorical variable.
    """

    n_unique = set(col.to_list())
    if mapping is None:
        mapping = {k: i for i, k in enumerate(n_unique, 0)}
    return col.map(mapping)


def one_hot_encoder(
    df: pd.DataFrame, cols: str | list, drop_first: bool = True
) -> pd.DataFrame:
    """
    Encode categorical variable as one-hot.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the categorical variable.
    col : pd.Series
        The categorical variable to be encoded.
    drop_first : bool, default=True
        Whether to drop the first column of the encoded variable.


    Returns:
    --------
    pd.DataFrame
        The encoded categorical variable.
    """
    if isinstance(cols, str):
        cols = [cols]

    return pd.get_dummies(df, columns=cols, drop_first=drop_first, dtype=int)


