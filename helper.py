from functools import partial, wraps
import pandas as pd
import polars as pl


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
