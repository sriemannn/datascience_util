
import pandas as pd

def target_window(
    data: pd.DataFrame, target: str, window: int = 3, dropna=True
):
    """
    Shift the target variable by a given window size for time series analysis

    Parameters:
    -----------
    data : pd.DataFrame
        The data to shift.
    target : str
        The target variable to shift.
    window : int, default=3
        The window size.

    Returns:
    --------
    pd.DataFrame : The shifted data.
    """
    shifts = [data[target].shift(i) for i in range(1, window + 1)]
    col_names = (f"{target}_t-{i}" for i in range(1, window + 1))

    data = data.assign(**dict(zip(col_names, shifts)))

    if dropna:
        return data.dropna()

    return data
