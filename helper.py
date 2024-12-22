from functools import partial, wraps
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.colors as mc


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
            key for key in ["df_train", "df_test", "target"] if key not in kwargs.keys()
        ]

        if missing_keys:
            raise KeyError(f"Missing keys in function call: {missing_keys}")

        df_train = kwargs["df_train"]
        df_test = kwargs["df_test"]
        target = kwargs["target"]

        df_type = "pandas" if isinstance(df_train, pd.DataFrame) else "polars"

        def drop(df, target):
            return (
                df.drop(columns=[target]) if df_type == "pandas" else df.drop([target])
            )

        y_train = df_train[target]
        X_train = drop(df_train, target)

        if target in df_test.columns:
            y_test = df_test[target]
            X_test = drop(df_test, target)
        else:
            X_test = df_test

        kwargs["df_train"] = X_train
        kwargs["df_test"] = X_test

        X_train, X_test = func(**kwargs)

        concat_fun = (
            partial(pd.concat, axis=1)
            if df_type == "pandas"
            else partial(pl.concat, how="horizontal")
        )

        df_test = concat_fun([X_test, y_test]) if target in df_test.columns else X_test
        df_train = concat_fun([X_train, y_train])

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

        colors = mc.LinearSegmentedColormap.from_list("custom", ["#FF0000", "#FFFFFF"])

        return df1.style.background_gradient(axis=None, gmap=mask_df, cmap=colors)

    # Filter out the outlier rows
    outliers_in_df1 = get_outliers(df1, df2, common_cols)
    outliers_in_df2 = get_outliers(df2, df1, common_cols)

    outliers_in_df1 = outliers_in_df1[common_cols]
    outliers_in_df2 = outliers_in_df2[common_cols]

    # Concatenate all outliers into a single dataframe
    return outliers_in_df1, outliers_in_df2
