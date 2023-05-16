# from numba import njit, prange
# from numba import njit
import numpy as np
import pandas as pd
# import talib as ta
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
"""
因子分析模块
"""


class FactorRanker():

    def __init__(self, factors: pd.DataFrame or pd.Series, bins: int,
                 sample_size: int):
        """
        本模块是根据因子与之前sample_siez期因子值按照从小到大排序, 分成样本数量一致的bins份, 然后根据排序确定该因子值的rank
        args:
        factors: 因子
        bins: 根据因子值把因子分成多少组
        sample_size: 每份用于排序的因子的样本数量
        """
        self.factors = factors
        self.bins = bins
        self.sample_size = sample_size

    def preprocess_factors(self):
        """
        对因子进行预处理
        """
        self.factors = self.factors.dropna()
        if type(self.factors.index) != pd.core.indexes.datetimes.DatetimeIndex:
            self.factors.set_index("datetime", inplace=True, drop=True)
        if type(self.factors.index) != pd.core.indexes.datetimes.DatetimeIndex:
            raise Exception("因子中的\"datetime\",请检查因子")

    @staticmethod
    def _cal_rank_parallel(column, factors, bins, sample_size, len_data):
        _ranks = [None for i in range(len_data)]
        if len_data < sample_size:
            raise ("数据量小于sample_size")
        for i in range(sample_size, len_data):
            _rank = pd.qcut(factors[column].iloc[i + 1 - sample_size:i + 1],
                            q=bins,
                            labels=False,
                            duplicates="drop").iloc[-1]
            _ranks[i] = _rank
        return column, _ranks

    def _cal_ranks(self, factors) -> pd.DataFrame:
        data = factors
        data.dropna(inplace=True)
        len_data = len(data)

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self._cal_rank_parallel, column, factors,
                                self.bins, self.sample_size, len_data): column
                for column in factors.columns
            }

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

    def rank_factors(self, symbol: str, save=False) -> pd.DataFrame:

        self.preprocess_factors()
        factors = self.factors
        rank_factors = self._cal_ranks(factors)
        if save is True:
            rank_factors.to_parquet(
                f".//data//{symbol}_{self.sample_size}_{self.bins}_rank_df.parquet"
            )
        return rank_factors

    def cal_returns(self,
                    prices: pd.Series,
                    n_minutes_list: list = list(range(1, 31)) +
                    list(range(30, 201, 5))):
        if type(prices.index) != pd.core.indexes.datetimes.DatetimeIndex:
            raise Exception("prices的index应设为\"DatetimeIndex\",以便于对齐factors")
        rtn = pd.DataFrame()
        for i in n_minutes_list:
            rtn["shift_{}_rtn".format(i)] = np.log(
                prices.shift(-i) / prices) * 100
        # rtn["price"] = prices
        return rtn


class RankFactorAnalyzer:
    """
    1.把因子分成n份,循环计算 该因子与之前sample_siez期因子值中, 该因子的rank
    2.根据因子的rank, 计算每份因子对应收益率的mean, win_rate
    """

    def __init__(self, rank_factors, returns):
        self.rank_factors = rank_factors
        self.returns = returns
        if type(self.rank_factors.index
                ) != pd.core.indexes.datetimes.DatetimeIndex:
            raise Exception("rank_factors的index应设为\"DatetimeIndex\"")
        if type(self.returns.index) != pd.core.indexes.datetimes.DatetimeIndex:
            raise Exception("returns的index应设为\"DatetimeIndex\"")
        self.rank_df = pd.concat([rank_factors, returns], axis=1).dropna()
        # print(self.rank_df)

    def _cal_effective_period_acf(self,
                                  rank_df,
                                  select_df,
                                  sort_by="mean_rtn",
                                  top_n=30,
                                  window=20,
                                  threshold=0.5):
        """
        summary:
        计算因子有效期, 以自相关系数小于threshold为有效期结束

        args:
        top_n: 选取排名前n的因子
        window: 计算因子有效期的滚动平均周期
        threshold: 计算因子有效期的自相关系数最低值, 当低于这个值时认定有效期结束
        """
        select_df.assign(effective_period=None)
        if top_n > len(select_df):
            raise ("top_n 大于 select_df的长度")
        if window > len(rank_df):
            raise ("window 大于 rank_df的长度")
        if threshold > 1:
            raise ("threshold 大于 1")
        if threshold < 0:
            raise ("threshold 小于 0")
        if sort_by not in select_df.columns:
            raise (
                "sort_by 不在 select_df的列中, 请在[\"mean_rtn\", \"win_rate\"]中选择")
        if top_n == "all":
            select_idx = select_df.index
        else:
            select_idx = select_df.index[0:top_n]
        select_df = select_df.sort_index()  # 对索引进行排序，以提高性能

        for fac, rank, rtn in select_idx:
            cols = ["{}".format(fac), rtn]
            query_df = rank_df[cols].query("{}== {} ".format(fac, rank))
            rolling_mean = query_df[rtn].rolling(window=window).mean()
            # 计算自相关系数
            lag = 3000
            correlations = acf(rolling_mean.dropna(), nlags=lag, fft=True)
            # 计算因子有效周期
            below_threshold_indice = np.where(correlations < threshold)[0]
            below_threshold_indice = below_threshold_indice[0] if len(
                below_threshold_indice) > 0 else 3000

            select_df.loc[(fac, rank, rtn),
                          "effective_period"] = below_threshold_indice

        select_df.sort_values(sort_by, ascending=False, inplace=True)

        return select_df

    def _cal_effective_period_mean(self,
                                   rank_df,
                                   select_df,
                                   sort_by="mean_rtn",
                                   top_n=30,
                                   window=20):
        """
            summary:
            计算因子有效期, 其后滚动收益率低于因子收益率均值的次数

            args:
            top_n: 选取排名前n的因子
            window: 计算因子有效期的滚动平均周期

            """
        select_df.assign(effective_period=None)
        select_idx = select_df.index[0:top_n]
        select_df = select_df.sort_index()  # 对索引进行排序，以提高性能

        for fac, rank, rtn in select_idx:
            cols = ["{}_rank".format(fac), rtn]
            # 求组rank_df中前3000个因子收益的中位数
            miu = select_df.loc[(fac, rank, rtn), "mean_rtn"].mean()
            query_df = rank_df[cols].query("{}_rank == {} ".format(fac, rank))
            rolling_mean = query_df.rolling(window).mean()
            rolling_mean.dropna(inplace=True)
            a = rolling_mean.copy().reset_index(drop=True)
            # 找到大于miu的索引
            greater_than_miu = a[a['short_liqka'] > miu].index
            # 遍历大于miu的索引并计算有效期数
            for idx in greater_than_miu:
                count = 1
                for i in range(idx + 1, len(a)):
                    if a.at[i, 'short_liqka'] > miu:
                        count += 1
                    else:
                        break
                a.at[idx, 'effective_period'] = count

            sample_mean = a["effective_period"][
                a["effective_period"] > 0].mean()

            select_df.loc[(fac, rank, rtn), "effective_period"] = sample_mean
            select_df.sort_values(sort_by, ascending=False, inplace=True)

        return select_df

    def cal_rank_results(
        self,
        save=False,
        symbol: str = None,
        bins: int = None,
        sample_size: int = None,
    ) -> pd.DataFrame:

        factors = self.rank_factors
        rtn = self.returns
        factors_cols = [col for col in factors.columns if "rank" in col]
        rtn_cols = [rtn for rtn in rtn.columns if "rtn" in rtn]

        rank_df = self.rank_df.copy()

        results_df = pd.DataFrame()
        count_df = pd.DataFrame()

        for i in factors_cols:
            count_df = rank_df.groupby(i)["open"].count().rename("counts")

            mean_rtn = rank_df.groupby(i)[rtn_cols].mean()

            win_rate = rank_df.groupby(i)[[
                rtn for rtn in rtn_cols
            ]].apply(lambda x: (x > 0).mean() * 100)

            result = pd.merge(mean_rtn,
                              win_rate,
                              on=i,
                              how="outer",
                              suffixes=("_mean",
                                        "_win_rate")).merge(count_df,
                                                            on=i,
                                                            how="outer")

            result.index = pd.MultiIndex.from_product([[i], result.index],
                                                      names=["factor", "rank"])
            results_df = pd.concat([results_df, result], axis=0)

        if save is True:
            results_df.to_parquet(
                f".//data//{symbol}_{sample_size}_{bins}_results_df.parquet")

        return results_df

    def factors_select(self,
                       results_df=None,
                       win_rate=0,
                       rtn=-1,
                       count=None,
                       sort_by="mean_rtn",
                       top_n=30,
                       threshold=0.5,
                       window=20):
        """
        根据条件选择因子:\n
        1.胜率大于win_rate且收益率大于rtn的因子;\n
        2.因子的样本数量大于count, 如果count不填写, 则默认为len(rank_df)//bins的一半, 如果一半小于500, 则默认为500;  \n
        args:\n
        results_df: 每个因子的rank对应收益率的mean, win_rate,effective_period;\n
        win_rate: 胜率，只选择胜率大于win_rate的因子;\n
        rtn: 收益率，只选择收益率大于rtn的因子;\n
        count: 样本数量，只选择样本数量大于count的因子;\n
        sorted: 排序方式,可选"win_rate", "rtn";\n
        top_n: 根据sorted选择的排序方式, 选择top_n个因子进行有效期计算, 如果要算全部的因子有效期,则设置为"all";\n
        threshold: 计算因子有效期的自相关系数最低值, 当低于这个值时认定有效期结束;\n
        window: 计算因子有效期的滚动平均周期;
        """
        df_idx = []
        mean_s = pd.Series(dtype=float)
        win_rate_s = pd.Series(dtype=float)
        count_s = pd.Series(dtype=int)
        effective_period_s = pd.Series(dtype=int)

        for col_name in results_df.columns:

            # 创建一个新的Series，其中包含该列的数据，其索引为三层MultiIndex
            if "mean" in col_name:
                _df_idx = [(idx[0], idx[1], col_name[:-5])
                           for idx in results_df.index]
                df_idx.extend(_df_idx)

                _mean_s = pd.Series(results_df[col_name].values)
                mean_s = pd.concat([mean_s, _mean_s], axis=0)
                _count_s = pd.Series(results_df["counts"].values)
                count_s = pd.concat([count_s, _count_s], axis=0)
            if "win_rate" in col_name:
                _win_rate_s = pd.Series(results_df[col_name].values)
                win_rate_s = pd.concat([win_rate_s, _win_rate_s], axis=0)
            if "effective_period" in col_name:
                _effective_period_s = pd.Series(results_df[col_name].values)
                effective_period_s = pd.concat(
                    [effective_period_s, _effective_period_s], axis=0)

        df = pd.concat([mean_s, win_rate_s, count_s, effective_period_s],
                       axis=1)
        df.index = pd.MultiIndex.from_tuples(df_idx,
                                             names=('factor', 'rank',
                                                    "return"))
        df.columns = ["mean_rtn", "win_rate", "count", "effective_period"]

        if count is None:
            count = 500
        select_df = df[(df["mean_rtn"] > rtn) & (df["win_rate"] > win_rate) &
                       (df["count"] > count)]
        select_df = select_df.sort_values(by=sort_by, ascending=False)
        # print(select_df)

        select_df = self._cal_effective_period_acf(self.rank_df,
                                                   select_df,
                                                   sort_by=sort_by,
                                                   top_n=top_n,
                                                   window=window,
                                                   threshold=threshold)
        return select_df

    def plots(
        self,
        results_df,
        n=3,  # 画出前n个因子的图
        show_dt=False,  # 画图是否显示时间信息-因为数据并非时间连续, 选True会导致图像不连续
        sorted="mean_rtn",  # 可以选择"mean_rtn"或者"win_rate"
        selected_args={
            "win_rate": 0,
            "rtn": -1,
            "count": None
        }):
        """
        画出因子的累计收益率图
        args:
            n: 画出前n个因子的图
            show_dt: 画图是否显示时间信息-因为数据并非时间连续, 选True会导致图像不连续
            sorted: 可以选择"mean_rtn"或者"win_rate"
            selected_args: 选择因子的条件, 默认为胜率大于50, 收益率大于0.1, 样本数量大于500
        """
        print("画出前{}个因子的图".format(n))
        rank_df = self.rank_df
        for k, v in selected_args.items():
            if k == "win_rate":
                win_rate = v
            if k == "rtn":
                rtn = v
            if k == "count":
                count = v

        seletced = self.factors_select(results_df,
                                       win_rate=win_rate,
                                       rtn=rtn,
                                       count=count)
        seletced = seletced.sort_values(sorted, ascending=False)
        w = seletced["win_rate"].values
        r = seletced["mean_rtn"].values
        factor = seletced.index.get_level_values(0)
        rank = seletced.index.get_level_values(1)
        rtn = seletced.index.get_level_values(2)

        for i in range(n):
            s = rank_df[rtn[i]][rank_df[factor[i]] == rank[i]]
            s = (s / 100 + 1).cumprod()
            s
            if show_dt is False:
                s.reset_index(inplace=True, drop=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(s)
            ax.set_ylabel("cumprod", fontsize=16)
            ax.set_title(
                f"{factor[i]}_{rank[i]} :{s.name}    win_rate:{round(w[i],2)}  mean_rtn:{round(r[i],3)}",
                fontsize=22)
            plt.show()
