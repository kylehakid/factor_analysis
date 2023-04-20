from numba import njit, prange
from numba import njit
import numpy as np
import pandas as pd
import talib as ta
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

"""
因子分析模块
"""


class FactorAnalysis_ori:
    """
    1.把因子分成n份,循环计算 该因子与之前sample_siez期因子值中, 该因子的rank
    2.根据因子的rank, 计算每份因子对应收益率的mean, win_rate
    """

    def __init__(self):
        self.factor_ranked_run = False

    @staticmethod
    def _cal_rank_parallel(column, factors, bins, sample_size, len_data):
        _ranks = [None for i in range(len_data)]
        if len_data < sample_size:
            raise ("数据量小于sample_size")
        for i in range(sample_size, len_data):
            _rank = pd.qcut(
                factors[column].iloc[i+1 - sample_size:i+1], q=bins, labels=False, duplicates="drop").iloc[-1]
            _ranks[i] = _rank
        return column, _ranks

    def _cal_ranks(self, factors, rtn) -> dict:
        data = pd.concat([self.factors, self.rtn], axis=1)
        data.dropna(inplace=True)
        len_data = len(data)

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self._cal_rank_parallel, column, factors,
                                       self.bins, self.sample_size, len_data): column for column in factors.columns}

            rank_list = []
            completed_count = 0
            total_count = len(futures)
            for future in as_completed(futures):
                completed_count += 1
                progress = completed_count / total_count * 100
                print(f"Progress: {progress:.2f}%", end="\r")
                rank_list.append(future.result())

        for column, _ranks in rank_list:
            data[f"{column}_rank"] = _ranks

        data.dropna(inplace=True)
        return data

    def factor_ranked(self, factors: pd.DataFrame, rtn: pd.DataFrame, bins: int = 15, sample_size=30000, save=False, symbol=None) -> pd.DataFrame:
        """
        根据该因子与之前sample_siez期因子值按照从小到大排序, 分成样本数量一致的bins份, 然后根据排序确定该因子值的rank
        计算每个rank因子对应收益率的mean, win_rate

        返回两个dataframe,rank_df: 每个因子的rank, results_df: 每个因子的rank对应收益率的mean, win_rate
        args:
        factors: 因子
        rtn: 收益率
        bins: 根据因子值把因子分成多少组
        sample_size: 每份样本数量
        """
        self.factors = factors
        self.rtn = rtn
        self.bins = bins
        self.sample_size = sample_size
        self.factor_ranked_run = True

        factors_col = self.factors.columns
        rtn_cols = self.rtn.columns

        results_df = pd.DataFrame()
        count_df = pd.DataFrame()
        if len(factors) != len(rtn):
            raise ValueError("factors and rtn must have the same length")

        # n = len(factors)
        # if n >= 10000:
        #     rank_df = pd.DataFrame()
        #     n_batches = int((n / (sample_size + 10000)))+1
        #     batch_size = (sample_size + 10000)
        #     print("数据太大,分成{}份计算".format(n_batches))

        #     for idx in range(n_batches):
        #         print("\n", "计算第{}份".format(idx+1))
        #         start_idx = idx * batch_size - idx*sample_size
        #         end_idx = min((idx + 1) * batch_size, n)
        #         factors_batch = factors.iloc[start_idx:end_idx]
        #         rtn_batch = rtn.iloc[start_idx:end_idx]

        #         _rank_df = self._cal_ranks(factors_batch, rtn_batch)
        #         rank_df = pd.concat([rank_df, _rank_df], axis=0)
        #         rank_df = rank_df.groupby(rank_df.index).first()

        # else:
        rank_df = self._cal_ranks(factors, rtn)

        for i in factors_col:
            count_df = rank_df.groupby(f"{i}_rank")["open"].count()
            count_df.rename("counts", inplace=True)
            mean_rtn = rank_df.groupby(f"{i}_rank")[
                [rtn for rtn in rtn_cols]].mean()
            win_rate = rank_df.groupby(f"{i}_rank")[
                [rtn for rtn in rtn_cols]].apply(lambda x: (x > 0).mean()*100)
            result = pd.merge(
                mean_rtn, win_rate, on=f"{i}_rank", how="outer", suffixes=("_mean", "_win_rate")).merge(count_df, on=f"{i}_rank", how="outer")
            # result.index.name = "rank"
            result.index = pd.MultiIndex.from_product(
                [[i], result.index], names=["factor", "rank"])
            results_df = pd.concat([results_df, result], axis=0)
            self.results = results_df

            if save == True:
                if os.path.exists("data"):
                    pass
                else:
                    os.mkdir("data")
                if symbol is not None:
                    rank_df.to_parquet(
                        f".//data//{symbol}_{bins}_rank_df.parquet")
                    results_df.to_parquet(
                        f".//data//{symbol}_{bins}_results_df.parquet")
                else:
                    symbol = "symbol"
                    rank_df.to_parquet(
                        f".//data//{symbol}_{bins}_rank_df.parquet")
                    results_df.to_parquet(
                        f".//data//{symbol}_{bins}_results_df.parquet")

        return rank_df, results_df

    def factors_select(self, results_df, win_rate=50, rtn=0.1, count=None):
        """
        根据条件选择满足
        1.胜率大于win_rate且收益率大于rtn的因子
        2.因子的样本数量大于count, 如果count不填写, 则默认为sample_size//bins的一半, 如果一半小于500, 则默认为500
        """
        df_idx = []
        mean_s = pd.Series(dtype=float)
        win_rate_s = pd.Series(dtype=float)
        count_s = pd.Series(dtype=int)
        if self.factor_ranked_run == True:
            sample_size = self.sample_size
            bins = self.bins
        else:
            sample_size = results_df.loc[results_df.index.get_level_values(0)[
                0], "counts"].sum()
            bins = len(results_df.index.get_level_values(1).unique())

        if count == None:
            count = max((sample_size//bins)*0.5, 500)
        # 遍历results_df的每一列
        for col_name in results_df.columns:
            # 创建一个新的Series，其中包含该列的数据，其索引为三层MultiIndex
            if "mean" in col_name:
                _df_idx = [(idx[0], idx[1], col_name)
                           for idx in results_df.index]
                df_idx.extend(_df_idx)

                _mean_s = pd.Series(results_df[col_name].values)
                mean_s = pd.concat([mean_s, _mean_s], axis=0)
                _count_s = pd.Series(results_df["counts"].values)
                count_s = pd.concat([count_s, _count_s], axis=0)
            if "win_rate" in col_name:
                _win_rate_s = pd.Series(results_df[col_name].values)
                win_rate_s = pd.concat([win_rate_s, _win_rate_s], axis=0)

        df = pd.concat([mean_s, win_rate_s, count_s], axis=1)
        df.index = pd.MultiIndex.from_tuples(
            df_idx, names=('factor', 'rank', "return"))
        df.columns = ["mean_rtn", "win_rate", "count"]
        df.sort_values(by="win_rate", inplace=True, ascending=False)

        df = df[(df["mean_rtn"] > rtn) & (
            df["win_rate"] > win_rate) & (df["count"] > count)]

        return df
