from numba import njit, prange
import concurrent.futures
from numba import njit
import numpy as np
import pandas as pd
import talib as ta
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from numba import njit
"""
适用于bardata的函数, 因子值尽量约束在[-1,1]
"""


def sma_diff(data, n, colums: str = "close") -> pd.Series or pd.DataFrame:
    """
    根据data.close计算n期的sma.返回sma的差值/当期close
    """
    sma = data[colums].rolling(n).mean()
    return (sma - data[colums]) / data[colums] * 100


def ema_diff(data, n, colums: str = "close") -> pd.Series or pd.DataFrame:
    """
    根据data.close计算n期的ema.返回ema的差值/当期close
    """
    ema = data[colums].ewm(span=n).mean()
    return (ema - data[colums]) / data[colums] * 100


def sma_of_sma(data, long, short, colums: str = "close") -> pd.Series or pd.DataFrame:
    """
    根据data.close计算n期的sma, 再计算m期的sma.返回m期sma的差值/当期close
    """
    sma = data[colums].rolling(short).mean()
    sma_sma = sma.rolling(long).mean()
    return (sma_sma - data[colums]) / data[colums] * 100


def sma_diff_sma(data, long, short, colums: str = "close") -> pd.Series or pd.DataFrame:
    """
    根据data.close计算n期的sma, 再计算m期的sma.返回m期sma的差值/当期close
    """
    sma1 = data[colums].rolling(long).mean()
    sma2 = sma1.rolling(short).mean()
    return (sma2-sma1)/sma1*100


def ema_of_ema(data, n, m, colums: str = "close") -> pd.Series or pd.DataFrame:
    """
    根据data.close计算n期的ema, 再计算m期的ema.返回m期ema的差值/当期close
    """
    ema = data[colums].ewm(span=n).mean()
    ema_ema = ema.ewm(span=m).mean()
    return (ema_ema - data[colums]) / data[colums] * 100


def high(data, n) -> pd.Series:
    """
    计算n期的high与当前值差多少期/总期数
    """

    high_index = data.close.rolling(n).apply(lambda x: np.argmax(x))
    return (n - high_index) / n


def low(data, n) -> pd.Series:
    """
    计算n期的low与当前值差多少期/总期数
    """
    low_index = data.close.rolling(n).apply(lambda x: np.argmin(x))
    return (n - low_index) / n


def wl(data, n) -> pd.Series:
    """
    计算n期的威廉指标
    """
    high = data.high.rolling(n).max()
    low = data.low.rolling(n).min()
    return (data.close - low) / (high - low)


def MACD(data, fast, slow, mid):
    """
    计算n期的MACD
    dif =  短线EMA - 长线EMA
    dea = EMA(diff, mid)   (diff的平滑移动平均线)
    hist = (diff - dea)  (柱状图)
    """
    dif, dea, hist = ta.MACD(data.close,
                             fastperiod=fast,
                             slowperiod=slow,
                             signalperiod=mid)
    signal = (dif - dea) / dea
    return dif, dea, hist, signal


def slope(data, n):
    """
    计算n期的斜率
    """
    return ta.LINEARREG_SLOPE(data.close, timeperiod=n)


def rsi(data, n):
    """
    计算n期的rsi
    """
    return ta.RSI(data.close, timeperiod=n)


def sar(data):
    """
    计算n期的sar, 考虑用一分钟数据, 降低了acceleration到常规的0.1
    """
    _sar = ta.SAR(data.high, data.low, acceleration=0.002, maximum=0.2)

    return (data.close - _sar) / _sar * 100


# ---------------------------------------------------------------------------------------------
def rwr(df: pd.DataFrame, n: int):
    rwr = (df['close'] - df['open']) / (df['high'] - df['low']+0.000000001)
    rwr_ma = ta.SMA(rwr, n)
    return rwr_ma


def aroon(df: pd.DataFrame, n: int):
    aroon_up, aroon_dn = ta.AROON(df['high'], df['low'], n)
    aroon = aroon_up - aroon_dn
    return aroon_up, aroon_dn, aroon


def tendstrength(df: pd.DataFrame, n: int):
    # 计算x列两两差值的绝对值
    diff_abs = df['close'].diff().abs()

    # 计算前30个'diff_abs'的和
    totalabs = diff_abs.rolling(window=n-1).sum()

    # 计算前30个'x'的最后一个和第一个的差
    first_x = df['close'].shift(n - 1)
    last_first_diff = df['close'] - first_x
    ts = last_first_diff / totalabs
    return ts


def boll(df: pd.DataFrame, n: int):
    boll_mid = ta.SMA(df["close"], n)
    boll_std = ta.STDDEV(df["close"], n)

    dist = df['close'] - boll_mid
    boll = dist / (2 * boll_std)

    return boll


def don(df: pd.DataFrame, n: int):
    don_up = ta.MAX(df['close'], n)
    don_down = ta.MIN(df['close'], n)
    don_mid = 0.5 * (don_up + don_down)

    don_dist = (don_up - don_down)
    don = (df['close'] - don_mid) / don_dist
    return don


def sf01(df: pd.DataFrame, n: int):
    avg = (df["open"] + df["high"] + df['low'] + df['close'])/4
    avg2low = 2 * avg - df['low']
    avg2high = 2 * avg - df['high']
    max_avg2low = ta.MAX(avg2low, n)
    min_avg2high = ta.MIN(avg2high, n)

    sf01_mid = (max_avg2low + min_avg2high)/2
    sf01_dis = (max_avg2low - min_avg2high)
    sf01 = (df['close'] - sf01_mid) / sf01_dis
    return sf01


def cor_vol(df: pd.DataFrame, n: int):
    cor_vol = ta.CORREL(df["volume"], df["close"], n)
    return cor_vol


def cor_oi(df: pd.DataFrame, n: int):
    doi = df["open_interest"].diff(1)
    doi = doi.fillna(0)
    cor_oi = ta.CORREL(doi, df["close"], n)
    return cor_oi


# ---------------------------------------------------
"""
如下为计算收益率的函数
"""

# 计算n期的收益率


def rtn_shift(data: pd.Series, n):
    """
    计算n期后的对数收益率,注意这里的n应该取负数
    """

    return (np.log(data.shift(n)/data) - 2/10000) * 100


@njit
def calculate_exit_prices_long(open_prices, low_prices, trs, delta_t, min_thre):
    exit_prices = np.empty(len(open_prices))

    for i in range(len(open_prices)):
        liqka = 1.0
        lowafterentry = low_prices[i]
        stop_loss = lowafterentry * (1.0 - trs * liqka)

        j = i
        while j < len(open_prices):
            if i != j:
                liqka = max(1.0 - delta_t * (j - i), min_thre)
                lowafterentry = max(low_prices[i:j])
                stop_loss = lowafterentry * (1.0 - trs * liqka)

            if j < len(low_prices) - 1 and low_prices[j] <= stop_loss:
                exit_price = open_prices[j + 1]
                exit_prices[i] = exit_price
                break

            if j >= len(low_prices) - 1:
                exit_prices[i] = open_prices[-1]
                break

            j += 1

    return exit_prices


def long_liqka(data: pd.DataFrame, trs=0.03, delta_t=0.003, min_thre=0.5):
    """
    用liqka计算多头收益率
    trs: 止损比例
    delta_t: 每个tick的liqka减少量
    min_thre: 最小的liqka
    用当期的open价格买入, 通过其后行情的low价格计算止损价格, 用止损价格卖出, 计算收益率

    """
    open_prices = data.open.values
    low_prices = data.low.values
    exit_prices = calculate_exit_prices_long(
        open_prices, low_prices, trs, delta_t, min_thre)
    return (np.log(exit_prices/open_prices) - 2/10000)*100  # 1/10000是手续费


@njit
def calculate_exit_prices(open_prices, high_prices, trs, delta_t, min_thre):
    exit_prices = np.empty(len(open_prices))

    for i in range(len(open_prices)):
        liqka = 1.0
        highafterentry = high_prices[i]
        stop_loss = highafterentry * (1.0 + trs * liqka)

        j = i
        while j < len(open_prices):
            if i != j:
                liqka = max(1.0 - delta_t * (j - i), min_thre)
                highafterentry = min(high_prices[i:j])
                stop_loss = highafterentry * (1.0 + trs * liqka)

            if (j < len(high_prices) - 1) and (high_prices[j] >= stop_loss):
                exit_price = open_prices[j + 1]
                exit_prices[i] = exit_price
                break

            if j >= len(high_prices) - 1:
                exit_prices[i] = open_prices[-1]
                break

            j += 1

    return exit_prices


def short_liqka(data: pd.DataFrame, trs=0.03, delta_t=0.003, min_thre=0.5):
    """
    用liqka模型计算空头止损的对数收益率

    trs: 止损比例
    delta_t: 每个tick的liqka减少量
    min_thre: 最小的liqka
    用当期open作为买入价格, 通过之后的high大于止损价格时, 以max(下一期的open,止损价格)作为卖出价格
    止损价格 = 入场后的close的最小价*(1+trs)*liqka
    liqka = max(liqka - delta_t*期数, min_thre)

    """
    open_prices = data.open.values
    high_prices = data.high.values
    exit_prices = calculate_exit_prices(
        open_prices, high_prices, trs, delta_t, min_thre)
    return (np.log(open_prices / exit_prices) - 2/10000) * 100  # 1/10000是手续费
