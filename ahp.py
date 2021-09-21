import pandas as pd


def ahp(df):
    """
    Define the AHP method
    :param df: preference matrix
    :return: ranking
    """
    return ((df / df.apply('sum', axis=0)).apply("sum", axis=1)).sort_values(ascending=True)
