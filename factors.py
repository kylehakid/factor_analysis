import numpy as np
import pandas as pd
import talib as ta
"""
适用于bardata的函数
"""


def sma_diff(data, n, colums: str = "close") -> pd.Series or pd.DataFrame:
    """
    根据data.close计算n期的sma.返回sma的差值/当期close    

    """
    sma = data[colums].rolling(n).mean()
    return (sma - data[colums]) / data[colums]


def ema_diff(data, n, colums: str = "close") -> pd.Series or pd.DataFrame:
    """
    根据data.close计算n期的ema.返回ema的差值/当期close
    """
    ema = data[colums].ewm(span=n).mean()
    return (ema - data[colums]) / data[colums]


def sma_of_sma(data, n, m, colums: str = "close") -> pd.Series or pd.DataFrame:
    """
    根据data.close计算n期的sma, 再计算m期的sma.返回m期sma的差值/当期close
    """
    sma = data[colums].rolling(n).mean()
    sma_sma = sma.rolling(m).mean()
    return (sma_sma - data[colums]) / data[colums]


def ema_of_ema(data, n, m, colums: str = "close") -> pd.Series or pd.DataFrame:
    """
    根据data.close计算n期的ema, 再计算m期的ema.返回m期ema的差值/当期close
    """
    ema = data[colums].ewm(span=n).mean()
    ema_ema = ema.ewm(span=m).mean()
    return (ema_ema - data[colums]) / data[colums]


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

    return (data.close - _sar) / _sar


"""
如下为计算收益率的函数
"""


def long_liqka(data: pd.DataFrame, trs=0.03, delta_t=0.003, min_thre=0.015):
    """
    用liqka模型计算多头止损的对数收益率

    trs: 止损比例
    delta_t: 每个tick的liqka减少量
    min_thre: 最小的liqka
    用当期open作为买入价格, 通过之后的low小于止损价格时, 以min(下一期的open,止损价格)作为卖出价格
    止损价格 = 入场后的close的最大价*(1-trs)*liqka
    liqka = max(liqka - delta_t*期数,min_thre)
    """
    exit_prices = []
    for i in range(len(data)):
        for j in range(i, len(data)):
            if i == j:
                liqka = 1
                lowafterentry = data.low.iloc[i]
                stop_loss = lowafterentry * (1 - trs * liqka)
            else:
                liqka = max(1 - delta_t * (j - i), min_thre)
                lowafterentry = max(data.low.iloc[i:j])  # 截止到上一期low的最大值
                stop_loss = lowafterentry * (1 - (trs * liqka))

            # 当期low小于止损价格, 以min(下一期的open,止损价格)作为卖出价格
            if j < len(data) - 1 and data.low.iloc[j] <= stop_loss:
                # print("第", i, "期", "第", j, "个")
                # print("入场价格", data.open[i], "当期最低", data.low.iloc[j],
                #       "止损价格", stop_loss, "下一期开盘", data.open.iloc[j+1], "pnl", stop_loss-data.open[i])
                exit_price = max(data.open.iloc[j + 1], stop_loss)
                exit_prices.append(exit_price)
                break

            # 最后一期还没有止损, 以最后一期open作为卖出价格
            if j >= len(data) - 1:
                exit_prices.append(data.open.iloc[-1])
                break

    return np.log(exit_prices / data.open) - 1 / 10000


def short_liqka(data: pd.DataFrame, trs=0.03, delta_t=0.003, min_thre=0.015):
    """
    用liqka模型计算空头止损的对数收益率

    trs: 止损比例
    delta_t: 每个tick的liqka减少量
    min_thre: 最小的liqka
    用当期open作为卖出价格, 通过之后的high大于止损价格时, 以max(下一期的open,止损价格)作为买入价格
    止损价格 = 入场后的close的最小价*(1+trs)*liqka
    liqka = max(liqka - delta_t*期数,min_thre)
    """

    exit_prices = []
    for i in range(len(data)):

        for j in range(i, len(data)):

            if i == j:
                liqka = 1
                highafterentry = data.high.iloc[i]
                stop_loss = highafterentry * (1 + trs * liqka)
            else:
                liqka = max(1 - delta_t * (j - i), min_thre)
                highafterentry = min(data.high.iloc[i:j])  # 截止到上一期high的最小值
                stop_loss = highafterentry * (1 + (trs * liqka))

            # 当期high大于止损价格, 以max(下一期的open,止损价格)作为买入价格
            if (j < len(data) - 1) and (data.high.iloc[j] >= stop_loss):
                # print("第", i, "期", "第", j, "个")
                # print("入场价格", data.open[i], "当期最高", data.high.iloc[j],
                #       "止损价格", stop_loss, "下一期开盘", data.open.iloc[j+1], "pnl", data.open[i]-stop_loss)
                exit_price = min(data.open.iloc[j + 1], stop_loss)
                exit_prices.append(exit_price)
                break

            # 最后一期还没有止损, 以最后一期open作为买入价格
            if j >= len(data) - 1:
                exit_prices.append(data.open.iloc[-1])
                break

    return np.log(data.open / exit_prices)
