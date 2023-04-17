import talib as ta
import numpy as np
from vpny_qmlt.trader.utility import ArrayManager
import pandas as pd
# from ..factors import MACD
from vpny_qmlt.trader.bargenerator import BarGenerator
from vpny_qmlt.trader.template import (
    Template,
    Signal,
    TickData,
    BarData,
    TradeData,
    OrderData,
    Offset,
    Direction,
)
from vpny_qmlt.trader.utility import Interval


def MACD(data, fast, slow, mid):
    """
    计算n期的MACD
    dif =  短线EMA - 长线EMA
    dea = EMA(diff, mid)   (diff的平滑移动平均线)
    hist = (diff - dea)  (柱状图)
    """
    dif, dea, hist = ta.MACD(data,
                             fastperiod=fast,
                             slowperiod=slow,
                             signalperiod=mid)
    signal = (dif - dea) / dea
    return dif, dea, hist, signal


class TestDL(Signal):
    def __init__(self, engine, signal_name, interval, windows, vt_symbols, setting):
        super().__init__(engine, signal_name, interval,
                         windows, vt_symbols, setting)

        self.am = ArrayManager(size=(self.sample_size))

        self.open_intenses = []

        # args

    def on_init(self):
        self.log_info("信号初始化")

        self.load_bar(days=300)
        self.subscribe_bar(
            self.vt_symbols, Interval.MINUTE, 1, self.on_1_bar)

    def on_1_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        open_intense = 0
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        dif, dea, hist, signal = MACD(self.am.close, fast=10, slow=60, mid=15)
        rank = pd.qcut(
            dea, q=self.bins, labels=False, duplicates="drop")[-1]
        # print(rank)
        if rank == 29:
            open_intense = 1
        self.update_intense(dt=bar.datetime, open_intense=open_intense)

    def on_1h_bar(self, bar: BarData):
        return

    def on_xh_bar(self, bar: BarData):
        return

# author = SJX


class Test01(Template):
    def __init__(self, engine, strategy_name: str, vt_symbols: list, setting: dict):
        # 参数
        super().__init__(engine, strategy_name, vt_symbols, setting)
        setting.values()
        self.signal1: Signal = self.add_signal(
            TestDL, "test", Interval.MINUTE, "self.min", self.vt_symbols, self.test)

        self.time_count = {}  # 记录订单编号及持仓时间
        self.local_ids = {}  # 记录订单编号
        self.cancle_count = 0

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.subscribe_bar(
            self.vt_symbols, Interval.MINUTE, 1, self.on_bar)
        self.log_info("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.log_info("策略启动")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        pass

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """

        # 当open_intense信号触发即以当前bar.close价格开单

        if self.inited():
            open_intense = self.get_open_intense()
            if open_intense > 0:
                self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.LONG,
                                offset=Offset.OPEN, price=bar.close * 1.01,
                                volume=1)

            tcopy = self.time_count.copy()
            for vt_orderid, timecount in tcopy.items():
                timecount += 1
                if timecount > 30 and vt_orderid in self.active_limit_orders:
                    self.cancel_order(vt_orderid)  # 挂单超过30分钟,撤单
                    self.cancle_count += 1
                    if self.cancel_count % 10 == 0:
                        print("cancel order count:{}".format(
                            self.cancle_count))
                    # print("cancel order:{}".format(vt_orderid))
                elif timecount > self.shift_rtn:
                    order: OrderData = self.get_order(vt_orderid)
                    if not order:
                        print("error")
                    else:
                        local_id = self.send_order(
                            order.vt_symbol, Direction.SHORT if order.direction is Direction.LONG else Direction.LONG, Offset.CLOSE, bar.close*0.99, order.traded)

                        self.time_count.pop(vt_orderid)
                        continue

                self.time_count[vt_orderid] = timecount

    def on_cancel(self, order: OrderData):
        """
        Callback of cancelled order
        """
        self.time_count.pop(order.vt_orderid, None)

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        if order.vt_orderid not in self.time_count and order.offset == Offset.OPEN:
            self.time_count[order.vt_orderid] = 0

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
