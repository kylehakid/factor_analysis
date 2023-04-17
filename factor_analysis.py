from numba import njit, prange
from numba import njit
import numpy as np
import pandas as pd
import talib as ta
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
因子分析模块
"""


class FactorAnalysis_ori:
    """
    1.把因子分成n份,循环计算 该因子与之前sample_siez期因子值中, 该因子的rank
    2.根据因子的rank, 计算每份因子对应收益率的mean, win_rate
    """

    def __init__(self):
        self.features_run = False

    @staticmethod
    def _cal_rank_parallel(column, factors, rtn, bins, sample_nums, len_data):
        _ranks = [None for i in range(len_data)]
        for i in range(0, len_data):
            if i >= sample_nums:
                _rank = pd.qcut(
                    factors[column].iloc[i - sample_nums:i], q=bins, labels=False, duplicates="drop").iloc[-1]
                _ranks[i] = _rank
        return column, _ranks

    def _cal_ranks(self) -> dict:
        data = pd.concat([self.factors, self.rtn], axis=1)
        data.dropna(inplace=True)
        len_data = len(data)

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self._cal_rank_parallel, column, self.factors, self.rtn,
                                       self.bins, self.sample_nums, len_data): column for column in self.factors.columns}

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

    def factor_ranked(self, factors: pd.DataFrame, rtn: pd.DataFrame, bins: int = 30, sample_nums=30000) -> pd.DataFrame:
        """
        根据该因子与之前sample_siez期因子值按照从小到大排序, 分成样本数量一致的bins份, 然后根据排序确定该因子值的rank
        计算每个rank因子对应收益率的mean, win_rate
        返回一个dataframe, 双重index为[factor, rank], columns为[**mean_rtn, **win_rate]
        args:
        factors: 因子
        rtn: 收益率
        bins: 根据因子值把因子分成多少组
        sample_nums: 每份样本数量
        """
        self.factors = factors
        self.rtn = rtn
        self.bins = bins
        self.sample_nums = sample_nums
        self.features_run = True

        factors_col = self.factors.columns
        rtn_col = self.rtn.columns

        results = pd.DataFrame()
        data = self._cal_ranks()

        def _win_rate(series):
            return (series > 0).mean()*100

        for i in factors_col:
            mean_rtn = data.groupby(f"{i}_rank")[
                [rtn for rtn in rtn_col]].mean()
            win_rate = data.groupby(f"{i}_rank")[
                [rtn for rtn in rtn_col]].apply(_win_rate)

            result = pd.merge(
                mean_rtn, win_rate, on=f"{i}_rank", how="outer", suffixes=("_mean", "_win_rate"))
            # result.index.name = "rank"
            result.index = pd.MultiIndex.from_product([[i], result.index])
            results = pd.concat([results, result], axis=0)
            results.index.names = ["factor", "rank"]
            self.results = results
        return results

    def returns_select(self, results, top=30):
        """
        返回一个dict, key为因子名, values为一个dataframe, 包含前top名的mean_rtn, win_rate
        results: factor_ranked的返回值

        """

        if self.features_run == True:
            results = self.results
        top10_rtn = {}
        for i in results.columns:
            if "mean" in i:
                j = i.replace("mean", "win_rate")
            if "win_rate" in i:
                j = i.replace("win_rate", "mean")
            _top10 = results.sort_values(
                i, ascending=False).head(top)[[i, j]]
            # print(_top10,type(_top10))

            top10_rtn[i] = _top10
        return top10_rtn

    def win_rate_select(self, top10_results: dict, win_rate=75):
        """
        选择大于win_rate>win_rate的因子
        top10_results: top10_returns的返回值
        """
        top10_dict = top10_results
        wr_df = pd.DataFrame()

        for key in top10_dict.keys():
            _df = top10_dict[key][top10_dict[key].iloc[:, 1] > win_rate]

            # _df的多重索引再增加一个索引,值为top10_dict[key].columns[1][1][:-9]
            return_name = top10_dict[key].columns[1][:-9]
            _df.set_index(pd.Index([return_name]*len(_df)),
                          append=True, inplace=True)
            _df.columns = ['mean', 'win_rate']
            wr_df = pd.concat([wr_df, _df], axis=0)
            wr_df.index.names = ['factor', 'rank', "returns"]
        wr_df = wr_df.sort_values(by="mean", ascending=False)
        return wr_df

    def batch_features(self, factors: pd.DataFrame, rtn: pd.Series, n=100) -> dict:
        """
        计算每份因子对应收益率的mean,win_rate,样本数
        返回一个dict, key为因子名,values为一个dataframe,包含mean_rtn,win_rate,样本数
        """
        self.factors = factors
        self.rtn = rtn
        self.n = n

        batch_size = len(self.factors)//self.n

        # 把factor和returns合并成一个dataframe, 避免数据错位
        data = pd.concat([self.factors, self.rtn], axis=1)
        data.dropna(inplace=True)

        columns = self.factors.columns
        print("features 的key:", columns)
        features = {}
        for i in columns:
            feature = {}
            for j in range(self.n):
                # 按照data[i]的值排序
                data.sort_values(by=i, inplace=True, ascending=True)
                # 选取data[i]的值在第j份的数据
                data_c = data.iloc[j *
                                   batch_size:(j+1)*batch_size].copy(deep=True)

                rtn = data_c.iloc[:, -1]
                if rtn.name != self.rtn.name:
                    raise ValueError("rtn格式有误")  # 确保rtn没选错

                win = len(data_c[rtn >= 0])
                win_rate = win/len(data_c)
                mean_rtn = rtn.mean()
                factor_range = [min(data_c[i]), max(data_c[i])]  # 因子值范围
                feature[j] = {"mean_rtn": mean_rtn, "win_rate": win_rate,
                              "factor_range": factor_range, "batch_size": batch_size}

                del data_c  # 释放内存

            # 把feature改为dataframe,其中index为key,columns为rtn,win_rate,factor_range,batch_size
            df_fe = pd.DataFrame.from_dict(feature, orient="index", columns=[
                "mean_rtn", "win_rate", "factor_range", "batch_size"])

            features[i] = df_fe

        self.features_run = True
        return features
