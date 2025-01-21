from functools import wraps

import pandas as pd


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

    
    Examples
    --------
    :: 

      @do_not_transform_target
      def some_processing_function(df_train, df_test, target):

        ...

        return df_train, df_test

      some_processing_function(df_train=df_train, df_test=df_test, target=target)
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

