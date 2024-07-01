from preprocessing import (check_outlier,cleaning_outlier)
from utils.constants import ERROR
import pandas as pd
import numpy as np
def run_checkoutlier(data:pd.Series):
    if not isinstance(data,pd.Series):
        print(f"{ERROR} WARNING")
        raise ValueError(f"tipe data harus Series")
    data= cleaning_outlier(data)
    print(data)
if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv('data/output.csv')
    # check value more than median
    check_outlier = check_outlier(df["exchange_rate"])
    