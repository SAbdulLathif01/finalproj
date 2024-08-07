import pandas as pd
import pytest

def test_example():
    assert 1 + 1 == 2

def test_dataframe():
    df = pd.read_csv("https://github.com/RaimbekovA/bank-card-fraud-detection-using-machine-learning/raw/main/CCFD%20v.2/creditcard.csv")
    # Check if the dataframe is not empty
    assert not df.empty, "Dataframe is empty"
