import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np


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
