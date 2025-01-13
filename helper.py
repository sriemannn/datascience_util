from functools import wraps

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def do_not_transform_target(func):
    """
    Decorator that makes sure that the target variable is not transformed.

    Whenever the target variable is present in the input DataFrame, it is removed before
    applying any transformation and added back to the DataFrame after the transformation is complete.

    The target variable might be missing from the test DataFrame.

    Parameters
    ----------
    df_train : pd.DataFrame
        Input DataFrame for training.
    df_test : pd.DataFrame
        Input DataFrame for testing.
    target : str
        Name of the target variable.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame for training.
    pd.DataFrame
        Modified DataFrame for testing.

    Raises
    ------
    KeyError
        If any of the keyword arguments (`df_train`, `df_test`, `target`) are missing.


    """

    @wraps(func)
    def wrapper(
        **kwargs,
    ):
        missing_keys = [
            key
            for key in ["df_train", "df_test", "target"]
            if key not in kwargs.keys()
        ]

        if missing_keys:
            raise KeyError(f"Missing keys in function call: {missing_keys}")

        df_train = kwargs["df_train"]
        df_test = kwargs["df_test"]
        target = kwargs["target"]

        y_train = df_train[target]
        X_train = df_train.drop(columns=[target])

        if target in df_test.columns:
            y_test = df_test[target]
            X_test = df_test.drop(columns=[target])
        else:
            X_test = df_test

        kwargs["df_train"] = X_train
        kwargs["df_test"] = X_test

        X_train, X_test = func(**kwargs)

        df_test = (
            pd.concat([X_test, y_test], axis=1)
            if target in df_test.columns
            else X_test
        )
        df_train = pd.concat([X_train, y_train], axis=1)

        return df_train, df_test

    return wrapper


def match_col_levels(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Checks that categorical columns (i.e., object and category types) have similar levels.

    For all category/object type columns check that if they appear in both data frames,
    that they have similar levels. If not return the outlier rows in a dataframe.

    Arguments
    ---------
    df1 (dp.DataFrame): First data frame
    df2 (dp.DataFrame): Second data frame

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]: Dataframes with outlier rows
    """

    df1 = df1.copy()
    df2 = df2.copy()

    cat_cols_df1 = df1.select_dtypes(include=["object", "category"]).columns
    cat_cols_df2 = df2.select_dtypes(include=["object", "category"]).columns

    # Find common categorical columns
    common_cols = list(set(cat_cols_df1).intersection(set(cat_cols_df2)))

    def get_outliers(df1, df2, common_cols):
        df1 = df1[common_cols].copy()
        df2 = df2[common_cols].copy()

        mask_df = df1.apply(lambda col: col.isin(df2[col.name].unique()))
        neg_mask_df = ~mask_df
        # df1 = df1[neg_mask_df.any(axis=1)]

        rows, _ = np.where(neg_mask_df)
        rows = np.unique(rows)

        df1 = df1.iloc[rows, :]
        neg_mask_df = neg_mask_df.iloc[rows, :]

        colors = mc.LinearSegmentedColormap.from_list(
            "custom", ["#FF0000", "#FFFFFF"]
        )

        return df1.style.background_gradient(
            axis=None, gmap=mask_df, cmap=colors
        )

    # Filter out the outlier rows
    outliers_in_df1 = get_outliers(df1, df2, common_cols)
    outliers_in_df2 = get_outliers(df2, df1, common_cols)

    # Concatenate all outliers into a single dataframe
    return outliers_in_df1, outliers_in_df2


def plot_missing_values(train, test):
    """
    Plot all missing values in the train and test sets.

    Parameters
    ----------
    train : pd.DataFrame
        Input DataFrame for training.
    test : pd.DataFrame
        Input DataFrame for testing.


    Returns
    -------
    None
    """

    _, ax = plt.subplots(1, 2, figsize=(9, 3))

    sns.heatmap(train.isna(), ax=ax[0], cmap="viridis", cbar=False)
    ax[0].set_title("Missing values in train set")

    sns.heatmap(test.isna(), ax=ax[1], cmap="viridis", cbar=False)
    _ = ax[1].set_title("Missing values in test set")

    plt.show()


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
