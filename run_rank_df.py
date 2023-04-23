import pandas as pd
from importlib import reload
import factor_analysis as fa
reload(fa)

data = pd.read_parquet(".//data//factors.parquet")
data.set_index("datetime", inplace=True, drop=True)
factors_cols = []
rtn_cols = []
for col in data.columns:
    if col not in ['datetime', 'trading_date', "symbol"]:
        if "rtn" not in col and "liqka" not in col:
            factors_cols.append(col)
        else:
            rtn_cols.append(col)
# data = data.head(11000)
factors = data[factors_cols]
rtn = data[rtn_cols]


def main():
    fal = fa.FactorAnalysis_ori()
    rank_df, results_df = fal.factor_ranked(
        factors, rtn, save=True, sample_size=60000, bins=20)

    rank_df, results_df = fal.factor_ranked(
        factors, rtn, save=True, sample_size=30000, bins=20)
    rank_df, results_df = fal.factor_ranked(
        factors, rtn, save=True, sample_size=20000, bins=15)
    rank_df, results_df = fal.factor_ranked(
        factors, rtn, save=True, sample_size=10000, bins=15)
    rank_df, results_df = fal.factor_ranked(
        factors, rtn, save=True, sample_size=5000, bins=10)
    rank_df, results_df = fal.factor_ranked(
        factors, rtn, save=True, sample_size=2000, bins=10)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()  # Add this line if you plan to create an executable from your script
    print("start")
    main()
    print("\nend")
