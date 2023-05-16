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


class TestDL(Signal):
    def __init__(self, engine, signal_name,  setting):
        super().__init__(engine, signal_name,   setting)

        self.am = ArrayManager(size=(self.sample_size))

        self.open_intenses = []
        self.tradingdate = None

        # args

    def on_init(self):
        self.log_info("信号初始化")

        self.load_bar(days=30)
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

        # if self.tradingdate != bar.trading_date:
        #     self.tradingdate = bar.trading_date
        #     self.log_info(f"new date {self.tradingdate}")
        sma = ta.SMA(self.am.close, 5)
        diff = (sma - self.am.close)/self.am.close*100
        rank = pd.qcut(
            diff, q=self.bins, labels=False, duplicates="drop")[-1]
        # print(rank)
        if rank == 1 or rank == 2 or rank == 0:
            open_intense = -1
        self.update_intense(dt=bar.datetime, open_intense=open_intense)

    def on_1h_bar(self, bar: BarData):
        return

    def on_xh_bar(self, bar: BarData):
        return

# author = SJX


class Test01(Template):
    def __init__(self, engine, strategy_name: str, setting: dict):
        # 参数
        super().__init__(engine, strategy_name, setting)
        setting.values()
        self.signal1: Signal = self.add_signal(
            TestDL, "test", self.vt_symbols, self.test)

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
        long_signal = False
        short_signal = False

        long_pos = self.long_pos[bar.vt_symbol]
        short_pos = self.short_pos[bar.vt_symbol]

        # self.log_info(f"long_pos {long_pos}, short_pos {short_pos}")

        if self.inited():
            # 没有仓位
            # if self.pos > 5 or self.pos < -5:
            #     raise Exception(self.pos)
            open_intense = self.get_open_intense()
            if open_intense > 0:
                long_signal = True
            elif open_intense < 0:
                short_signal = True
            if -1000000 <= self.pos <= 3000000:  # 限制持仓数量
                if not self.get_at_risk():
                    if open_intense > 0:
                        long_signal = True

                    elif open_intense < 0:
                        short_signal = True

                    if not self.active_limit_orders:
                        if (long_signal is True):
                            self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.LONG,
                                            offset=Offset.OPEN, price=bar.close,
                                            volume=1)
                            # self.log_info(f'多开单,price:{bar.close}')
                        elif (short_signal is True):
                            self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.SHORT,
                                            offset=Offset.OPEN, price=bar.close,
                                            volume=1)
                            # self.log_info(f"多空单,price:{bar.close}")

                    else:
                        d1 = self.active_limit_orders.copy()
                        for orderid, order in d1.items():
                            if order.offset == Offset.OPEN:
                                # 多开撤单
                                if order.direction == Direction.LONG:
                                    if bar.close - order.price > 5:
                                        # self.log_info(
                                        #     f'多开撤单,high:{bar.high} - price:{order.price} > 5')
                                        self.cancel_order(orderid)
                                        # 撤单后再开仓
                                        # if long_signal:
                                        #     self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.LONG,
                                        #                     offset=Offset.OPEN, price=bar.close * 1.01,
                                        #                     volume=1)
                                # 空开撤单
                                else:
                                    if order.price - bar.close > 5:
                                        # self.log_info(
                                        #     f'空开撤单,price:{order.price} - low:{bar.low} > 5')
                                        self.cancel_order(orderid)
                                        # 撤单后再开仓
                                        # if short_signal:
                                        #     self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.SHORT,
                                        #                     offset=Offset.OPEN, price=bar.close*0.99,
                                        #                     volume=1)

            # 有仓位,平仓逻辑
            if long_pos or short_pos:
                close_pos, D_KliqPoint = self.get_liqka_close_intense(
                    bar.symbol)
                # self.log_info(
                #     f"close_pos{close_pos},{D_KliqPoint}, {self.long_pos},{self.short_pos}")

                if close_pos > 0:
                    # Long close
                    self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.LONG,
                                    offset=Offset.CLOSE, price=int(
                                        max(bar.open, D_KliqPoint)),
                                    volume=close_pos)
                elif close_pos < 0:
                    self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.SHORT,
                                    offset=Offset.CLOSE, price=int(
                                        min(bar.open, D_KliqPoint)),
                                    volume=-close_pos)
            else:
                prc_tick = self.get_pricetick(bar.vt_symbol)
                d1 = self.active_limit_orders.copy()
                for orderid, order in d1.items():
                    if (order.offset != Offset.OPEN):
                        if order.direction == Direction.SHORT and order.price - bar.low > 5 * prc_tick:
                            # self.log_info(
                            #     f'空平撤单,price:{order.price} - low:{bar.low} > 5')
                            self.reclose_price = bar.low - 2 * prc_tick
                            self.cancel_order(orderid)
                        elif order.direction == Direction.LONG and bar.high - order.price > 5 * prc_tick:
                            # self.log_info(
                            #     f'多平撤单,high:{bar.high} - price:{order.price} > 5')
                            self.reclose_price = bar.high + 2 * prc_tick
                            self.cancel_order(orderid)

    def on_cancel(self, order: OrderData):
        """
        Callback of cancelled order
        """
        self.time_count.pop(order.vt_orderid, None)

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        # if order.offset == Offset.CLOSE:
        #     self.log_info(f"close order {order.direction }")
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        # self.log_info(f"on_trade,{trade.offset}")
        ...
