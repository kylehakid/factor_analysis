# from numba import njit, prange
# from numba import njit
import numpy as np
import pandas as pd
# import talib as ta
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import StandardScaler
from IPython.display import display

"""
因子分析模块
"""


class FactorRanker():

    def __init__(self,
                 factors: pd.DataFrame,
                 prices: pd.DataFrame or pd.Series,
                 normalize=False,
                 z_score_filter=20):
        """
        本模块是根据因子与之前sample_siez期因子值按照从小到大排序, 分成样本数量一致的bins份, 然后根据排序确定该因子值的rank
        args:
        factors: dataframe, 因子值, 需要index是datetime或者有一列是datetime;
        price: 用于计算收益率的价格, 需要index是datetime或者有一列是datetime;
        bins: 根据因子值把因子分成多少组
        sample_size: 每份用于排序的因子的样本数量
        normalize: 是否对因子值做标准化处理
        z_score_filter: 对收益率数据做去极值处理, 如果是None则不做处理
        """
        self.factors = factors
        self.prices = prices
        self.normalize = normalize
        self.z_score_filter = z_score_filter
        self.data_processd = False

    def _preprocess_data(self):
        """
        对因子进行预处理
        """
        self.factors = self.factors.dropna()

        if type(self.factors.index) != pd.core.indexes.datetimes.DatetimeIndex:
            self.factors.set_index("datetime", inplace=True, drop=True)
        if type(self.factors.index) != pd.core.indexes.datetimes.DatetimeIndex:
            raise Exception("因子中没有\"datetime\",请检查因子")

        if self.normalize:
            factors = self.factors.copy()
            scaler = StandardScaler()
            for column in factors.columns:
                factors[column] = scaler.fit_transform(
                    factors[column].values.reshape(-1, 1))
            self.factors = factors

        if self.z_score_filter is not None:
            forward_returns = self.prices.shift(-1) / self.prices - 1
            mask = abs(forward_returns - forward_returns.mean()) < (
                    self.z_score_filter * forward_returns.std())

            self.prices = self.prices[mask]

        self.factors.dropna(inplace=True)
        self.prices.dropna(inplace=True)

        self.data_processd = True

    @staticmethod
    def _cal_rank_parallel(column, factors, bins, sample_size, len_data):
        _ranks = [None for i in range(len_data)]
        if len_data < sample_size:
            raise Exception(f"数据量({len_data})小于sample_size({sample_size})")
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

        # 创建一个字典来收集所有的列
        new_cols = {}
        for column, _ranks in rank_list:
            new_cols[f"{column}_rank"] = _ranks
        new_cols = pd.DataFrame(new_cols, index=data.index)

        # 使用pd.concat一次性添加所有的列
        data = pd.concat([data, new_cols], axis=1, join="inner")
        # data.set_index("datetime", drop=True,inplace=True)
        data.dropna(inplace=True)
        return data

    def _cal_returns(self,
                     periods=None):
        prices = self.prices
        if type(prices.index) != pd.core.indexes.datetimes.DatetimeIndex:
            raise Exception("prices的index应设为\"DatetimeIndex\",以便于对齐factors")
        _rtn_dict = {}
        for i in periods:
            future_rtn = ((prices.shift(-i) - prices) / prices) * 100
            future_exit_dt = prices.index.to_series().shift(-i)

            # 把计算出的数据存到字典中
            _rtn_dict["future_{}_rtn(%)".format(i)] = future_rtn
            _rtn_dict["future_{}_exit_dt".format(i)] = future_exit_dt

        # 一次性创建 DataFrame，并保留原有的index
        rtn = pd.DataFrame(_rtn_dict, index=prices.index)
        return rtn

    def rank_factors(self,
                     symbol: str,
                     bins: int = 20,
                     sample_size: int = 3000,
                     save=False) -> pd.DataFrame:
        self.bins = bins
        self.sample_size = sample_size
        if not self.data_processd:
            self._preprocess_data()
        factors = self.factors
        rank_factors = self._cal_ranks(factors)
        if save is True:
            rank_factors.to_parquet(
                f".//data//{symbol}_{self.sample_size}_{self.bins}_rank_df.parquet"
            )
        return rank_factors

    def cal_returns(self,
                    symbol: str,
                    periods=None,
                    save=False) -> pd.DataFrame:
        """
        计算收益率
        """
        if periods is None:
            periods = list(range(1, 31)) + \
                      list(range(31, 201, 5)) + [300, 400, 500]
        if not self.data_processd:
            self._preprocess_data()
        returns = self._cal_returns(periods)

        if save is True:
            returns.to_parquet(
                f".//data//{symbol}_{self.sample_size}_{self.bins}_returns_df.parquet"
            )
        return returns

    def cal_factors_and_rtns(self,
                             symbol: str,
                             bins: int = 20,
                             sample_size: int = 3000,
                             periods=None,
                             save=False):
        if periods is None:
            periods = list(range(1, 31)) + \
                      list(range(30, 201, 5)) + [300, 400, 500]
        rank_df = self.rank_factors(symbol=symbol,
                                    bins=bins,
                                    sample_size=sample_size,
                                    save=False)
        returns = self.cal_returns(symbol, periods, save=False)
        result_df = pd.concat([rank_df, returns], axis=1, join="inner")
        result_df.dropna(inplace=True)
        if save:
            result_df.to_parquet(
                f".//data//{symbol}_{self.sample_size}_{self.bins}_rank_returns_df.parquet"
            )
        return result_df


def _cal_effective_period_mean(rank_df,
                               select_df,
                               sort_by="mean_rtn",
                               top_n=30,
                               window=10):
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


def _cal_effective_period_acf(rank_df,
                              select_df,
                              sort_by="mean_rtn",
                              top_n=30,
                              window=10,
                              threshold=0.1):
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
        raise Exception("top_n 大于 select_df的长度")
    if window > len(rank_df):
        raise Exception("window 大于 rank_df的长度")
    if threshold > 1:
        raise Exception("threshold 大于 1")
    if threshold < 0:
        raise Exception("threshold 小于 0")
    if sort_by not in select_df.columns:
        raise Exception(
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


class RankFactorAnalyzer:
    """
    1.把因子分成n份,循环计算该因子与之前sample_siez期因子值中该因子的rank
    2.根据因子的rank, 计算每份因子对应收益率的mean, win_rate
    """

    def __init__(self, rank_factors: pd.DataFrame, returns: pd.DataFrame):
        if not isinstance(rank_factors, pd.DataFrame) or not isinstance(returns, pd.DataFrame):
            raise TypeError("rank_factors and returns should be pandas DataFrame")
        if not isinstance(rank_factors.index, pd.core.indexes.datetimes.DatetimeIndex):
            raise Exception("rank_factors的index应设为\"DatetimeIndex\"")
        if not isinstance(returns.index, pd.core.indexes.datetimes.DatetimeIndex):
            raise Exception("returns的index应设为\"DatetimeIndex\"")

        self.cal_results = False
        self.cal_select_results = False
        self.select_long = pd.DataFrame()
        self.select_short = pd.DataFrame()
        rtn_cols = [rtn for rtn in returns.columns if "rtn" in rtn]
        exit_cols = [rtn for rtn in returns.columns if "exit" in rtn]
        if len(rtn_cols) != len(exit_cols):
            raise Exception("returns中收益率和出场时间不匹配")
        # 确保数据对齐
        self.data: pd.DataFrame = pd.concat([rank_factors, returns], axis=1,
                                            join="inner").dropna()
        self.rank_factors = self.data[[rank for rank in self.data.columns if "rank" in rank]]
        self.returns = self.data[[rtn for rtn in self.data.columns if "rtn" in rtn or "exit" in rtn]]

    def cal_rank_results(
            self,
            factors_df: pd.DataFrame = None,
            rtn_df: pd.DataFrame = None,
            save=False,
            symbol: str = None,
            bins: int = None,
            sample_size: int = None,
    ) -> pd.DataFrame:
        if factors_df is None:
            factors = self.rank_factors
        else:
            factors = factors_df
        if rtn_df is None:
            rtn = self.returns
        else:
            rtn = rtn_df
        if not isinstance(factors, pd.DataFrame) or not isinstance(rtn, pd.DataFrame):
            raise TypeError("rank_factors and returns should be pandas DataFrame")
        if not isinstance(factors.index, pd.core.indexes.datetimes.DatetimeIndex):
            raise Exception("rank_factors的index应设为\"DatetimeIndex\"")
        if save:
            if symbol is None:
                raise Exception("symbol is None")
            if bins is None:
                raise Exception("bins is None")
            if sample_size is None:
                raise Exception("sample_size is None")

        factors_cols = [col for col in factors.columns if "rank" in col]
        rtn_cols = [rtn for rtn in rtn.columns if "rtn" in rtn]

        data_df: pd.DataFrame = pd.concat([factors, rtn], axis=1,
                                          join="outer")
        results_df = pd.DataFrame()

        for i in factors_cols:
            # 计算每个分组的收益率均值, 胜率, 样本数, 如果数据有nan, 则忽略nan
            na_count = data_df.groupby(i)[rtn_cols].apply(lambda x: x.isnull().sum())
            not_na_count = data_df.groupby(i)[rtn_cols].count() - na_count
            mean_rtn = data_df.groupby(i)[rtn_cols].sum() / not_na_count
            win_rate = data_df.groupby(i)[rtn_cols].apply(lambda x: (x > 0).sum() * 100) / not_na_count
            count_df = data_df.groupby(i)[i].count().rename("counts")
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

        self.cal_results = True
        self.results_df = results_df

        return results_df

    def factors_select(self,
                       win_rate=0,
                       rtn=-1,
                       count=None,
                       sort_by="mean_rtn",
                       top_n=30,
                       threshold=0.1,
                       window=5,
                       factors_df: pd.DataFrame = None,
                       rtn_df: pd.DataFrame = None,
                       results_df=None,
                       show_result=True) -> tuple:
        """
        根据条件选择因子:\n
        1.胜率大于win_rate且收益率大于rtn的因子;\n
        2.因子的样本数量大于count, 如果count不填写, 则默认为len(data_df)//bins的一半, 如果一半小于500, 则默认为500;  \n
        args:\n
        results_df: 每个因子的rank对应收益率的mean, win_rate,effective_period;\n
        win_rate: 胜率，只选择胜率大于win_rate的因子;\n
        rtn: 收益率，只选择收益率大于rtn的因子;\n
        count: 样本数量，只选择样本数量大于count的因子;\n
        sorted: 排序方式,可选"win_rate", "rtn";\n
        top_n: 根据sorted选择的排序方式, 选择top_n个因子进行有效期计算, 如果要算全部的因子有效期,则设置为"all";\n
        threshold: 计算因子有效期的自相关系数最低值, 当低于这个值时认定有效期结束;\n
        window: 计算因子有效期的滚动平均周期;
        results_df: pd.DataFrame,因子rank对应收益率的mean, win_rate,effective_period;
        """
        if not 0 <= win_rate <= 100:
            raise Exception("win_rate应在0到100之间")
        if not isinstance(rtn, (int, float)):
            raise Exception("rtn应在-1到1之间")
        if count is not None and not isinstance(count, int):
            raise Exception("count应为整数类型")
        if sort_by not in ["mean_rtn", "win_rate"]:
            raise Exception("sort_by应为\"mean_rtn\"或\"win_rate\"")
        if not isinstance(top_n, int) or top_n <= 0:
            raise Exception("top_n应为正整数")
        if not 0 <= threshold <= 1:
            raise Exception("threshold应在0到1之间")
        if not isinstance(window, int) or window <= 0:
            raise Exception("window应为正整数")
        if factors_df is None:
            factors_df = self.rank_factors
        if rtn_df is None:
            rtn_df = self.returns

        df_idx = []
        mean_s = pd.Series(dtype=float)
        win_rate_s = pd.Series(dtype=float)
        count_s = pd.Series(dtype=int)
        effective_period_s = pd.Series(dtype=int)
        if self.cal_results is False and results_df is None:
            results_df = self.cal_rank_results(factors_df, rtn_df, save=False)
        elif self.cal_results is True and results_df is None:
            results_df = self.results_df

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
        select_long = df[(df["mean_rtn"] > rtn) & (df["win_rate"] > win_rate) &
                         (df["count"] > count)]
        select_long = select_long.sort_values(by=sort_by, ascending=False)

        select_long = _cal_effective_period_acf(self.data,
                                                select_long,
                                                sort_by=sort_by,
                                                top_n=top_n,
                                                window=window,
                                                threshold=threshold)
        select_short = df.copy()
        select_short["mean_rtn"] = -select_short["mean_rtn"]
        select_short["win_rate"] = 100 - select_short["win_rate"]
        select_short = select_short.sort_values(by=sort_by, ascending=False)
        select_short = _cal_effective_period_acf(self.data,
                                                 select_short,
                                                 sort_by=sort_by,
                                                 top_n=top_n,
                                                 window=window,
                                                 threshold=threshold)

        self.cal_select_results = True
        self.select_long = select_long
        self.select_short = select_short
        if show_result:
            print("long factor:")
            display(select_long)
            print("short factor:")
            display(select_short)
        return select_long, select_short

    def factors_select_daily(self,
                             rolling_days: int = 5,
                             sort_by: str = "mean_rtn",
                             top_n=5,
                             save=False,
                             symbol=None):
        """
        每日选出因子
        :param rolling_days: 根据过去rolling_days天的数据选出因子
        :param sort_by: 根据何种指标选择因子，'mean_rtn'或者'win_rate'
        :param top_n: 选择前top_n个因子
        :return: pd.DataFrame,每日选出的因子
        """
        if rolling_days <= 0 or top_n <= 0:
            raise Exception("rolling_days 和 top_n 都应为正整数")

        if sort_by not in ["mean_rtn", "win_rate"]:
            raise Exception("sort_by应为\"mean_rtn\"或\"win_rate\"")
        if save:
            if symbol is None:
                raise Exception("symbol不能为空")

        factors_daily_long = {}
        factors_daily_short = {}
        _temp_rtn = {}
        rolling_days = 20
        date = np.unique(self.returns.index.date)
        date = np.sort(date)
        rtn_cols = [col for col in self.returns.columns if "rtn" in col]
        for day in range(len(date) - rolling_days):
            for col in rtn_cols:
                start = date[day]
                if day + rolling_days < len(date):
                    end = date[day + rolling_days]
                    if (day + rolling_days + 1) < len(date):
                        _now = date[day + rolling_days + 1]  # 选出因子的日期是可用数据的第二天
                    else:
                        _now = date[day + rolling_days] + pd.Timedelta(days=1)
                else:
                    raise Exception("error")
                    break
                exit_dt: str = col[:-6] + "exit_dt"
                _temp_rtn[col] = self.returns[col][
                    (start <= self.returns[exit_dt].dt.date) & (self.returns[exit_dt].dt.date <= end)]
            _temp_rtns = pd.DataFrame(_temp_rtn)
            _temp_factors = self.rank_factors[
                (start <= self.data[exit_dt].dt.date) & (self.data[exit_dt].dt.date <= end)]

            self.cal_results = False
            _select_long, _select_short = self.factors_select(
                factors_df=_temp_factors,
                rtn_df=_temp_rtns,
                sort_by=sort_by,
                top_n=top_n,
                show_result=False)
            _select_long = _select_long[:top_n]
            _select_short = _select_short[:top_n]
            factors_daily_long[_now] = _select_long
            factors_daily_short[_now] = _select_short

        factors_daily_short = pd.concat(factors_daily_short, names=["date"])
        factors_daily_long = pd.concat(factors_daily_long, names=["date"])

        print("long factor:")
        display(factors_daily_long)
        print("short factor:")
        display(factors_daily_short)
        if save:
            factors_daily_long.to_parquet(f".//data//{symbol}_daily_select_long_factors.parquet")
            factors_daily_short.to_parquet(f".//data//{symbol}_daily_select_short_factors.parquet")
            print(f"long factors saved to .//data//{symbol}_daily_select_factors.parquet")

        self.cal_results = False
        return factors_daily_long, factors_daily_short

    def plots(
            self,
            n=3,  # 画出前n个因子的图
            show_dt=False,  # 画图是否显示时间信息-因为数据并非时间连续, 选True会导致图像不连续
            sort_by="mean_rtn",  # 可以选择"mean_rtn"或者"win_rate"
            selected_args={"win_rate": 0, "rtn": -1, "count": None}):
        """
        画出因子的累计收益率图
        args:
            n: 画出前n个因子的图
            show_dt: 画图是否显示时间信息-因为数据并非时间连续, 选True会导致图像不连续
            sorted: 可以选择"mean_rtn"或者"win_rate"
            selected_args: 选择因子的条件, 默认为胜率大于50, 收益率大于0.1, 样本数量大于500,仅当没有运行过factors_select时有效
        """
        print("画出前{}个因子的图".format(n))
        data_df = self.data
        win_rate = 0
        rtn = -1
        count = None
        for k, v in selected_args.items():
            if k == "win_rate":
                win_rate = v
            if k == "rtn":
                rtn = v
            if k == "count":
                count = v
        if self.cal_results is False:
            results_df = self.cal_rank_results()
        else:
            results_df = self.results_df

        if self.cal_select_results is False:
            select_long, select_short = self.factors_select(win_rate=win_rate,
                                                            rtn=rtn,
                                                            count=count,
                                                            sort_by=sort_by,
                                                            results_df=results_df
                                                            )
        else:
            select_long = self.select_long
            select_short = self.select_short

        w = select_long["win_rate"].values
        r = select_long["mean_rtn"].values
        factor = select_long.index.get_level_values(0)
        rank = select_long.index.get_level_values(1)
        rtn = select_long.index.get_level_values(2)

        print("long factor:")
        for i in range(n):
            # log_rtn = data_df[rtn[i]][data_df[factor[i]] == rank[i]]
            # log_rtn = log_rtn.cumsum()
            rtns = data_df[rtn[i]][data_df[factor[i]] == rank[i]]
            cum_rtn = rtns.cumsum()
            if show_dt is False:
                cum_rtn.reset_index(inplace=True, drop=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(cum_rtn)
            ax.set_ylabel("cumsum", fontsize=16)
            ax.set_title(
                f"{factor[i]}_{rank[i]} :{cum_rtn.name}    win_rate:{round(w[i], 2)}  mean_rtn:{round(r[i], 3)}",
                fontsize=22)
            plt.show()

        w = select_short["win_rate"].values
        r = select_short["mean_rtn"].values
        factor = select_short.index.get_level_values(0)
        rank = select_short.index.get_level_values(1)
        rtn = select_short.index.get_level_values(2)

        print("short factor:")
        for i in range(n):
            # log_rtn = -data_df[rtn[i]][data_df[factor[i]] == rank[i]]
            # log_rtn = log_rtn.cumsum()
            rtns = -data_df[rtn[i]][data_df[factor[i]] == rank[i]]
            cum_rtn = rtns.cumsum()
            if show_dt is False:
                cum_rtn.reset_index(inplace=True, drop=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(cum_rtn)
            ax.set_ylabel("cumsum", fontsize=16)
            ax.set_title(
                f"{factor[i]}_{rank[i]} :{cum_rtn.name}    win_rate:{round(w[i], 2)}  mean_rtn:{round(r[i], 3)}",
                fontsize=22)
            plt.show()
