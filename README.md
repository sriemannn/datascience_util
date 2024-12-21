# Data Science Util

Some useful functions for data science.

## Decorators

A decorator that makes sure that the target variable is not transformed.
Expects the function decorated to have keyword arguments `df_train`, `df_test`,
and `target`.

```python

@do_not_transform_target
def some_processing_function(df_train, df_test, target):

  ...

  return df_train, df_test
```

##
