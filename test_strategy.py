import talib as ta
import numpy as np
from vpny_qmlt.trader.utility import ArrayManager
# from vpny_qmlt.trader.bargenerator import BarGenerator
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
    def __init__(self, engine, signal_name, interval, windows, vt_symbols, setting):
        super().__init__(engine, signal_name, interval,
                         windows, vt_symbols, setting)

        self.am = ArrayManager(size=1000)

        self.open_intenses = []

    def on_init(self):
        self.log_info("信号初始化")

        self.load_bar(days=5)
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
        _sar = ta.SAR(self.am.high, self.am.low,
                      acceleration=self.acce, maximum=0.2)[-1]
        sar = (bar.close - _sar)
        # 多开,上涨趋势
        if sar >= 0:
            open_intense = 1
        # 空开,下降趋势
        elif sar < 0:
            open_intense = -1
        self.update_intense(dt=bar.datetime, open_intense=open_intense)

    def on_1h_bar(self, bar: BarData):
        return

    def on_xh_bar(self, bar: BarData):
        return

#author = SJX


class Test01(Template):
    def __init__(self, engine, strategy_name: str, vt_symbols: list, setting: dict):
        # 参数
        super().__init__(engine, strategy_name, vt_symbols, setting)
        setting.values()
        self.signal1: Signal = self.add_signal(
            TestDL, "test", Interval.MINUTE, "self.min", self.vt_symbols, self.test)

        self.highafterentry = 0
        self.lowafterentry = 0
        self.reclose_price = 0

        self.liQKA = 0
        self.DliqPoint = 0
        self.KliqPoint = 0

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
        # self.open_intense()
        # return
        if self.inited():
            # 没有仓位
            if self.pos > 1 or self.pos < -1:
                raise Exception(self.pos)
            if self.pos == 0:
                if not self.active_limit_orders:
                    if not self.get_at_risk():
                        open_intense = self.get_open_intense()
                        # 在此策略中，等价于   self.open_intense() self.open_intense(start = 0)
                        if open_intense > 0:
                            self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.LONG,
                                            offset=Offset.OPEN, price=bar.open,
                                            volume=1)
                        elif open_intense < 0:
                            self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.SHORT,
                                            offset=Offset.OPEN, price=bar.open,
                                            volume=1)
                else:
                    d1 = self.active_limit_orders.copy()
                    for orderid, order in d1.items():
                        if order.offset == Offset.OPEN:
                            # 多开撤单
                            if order.direction == Direction.LONG:
                                if bar.high - order.price > 5:
                                    # self.log_info(
                                    #     f'多开撤单,high:{bar.high} - price:{order.price} > 5')
                                    self.cancel_order(orderid)
                            # 空开撤单
                            else:
                                if order.price - bar.low > 5:
                                    # self.log_info(
                                    #     f'空开撤单,price:{order.price} - low:{bar.low} > 5')
                                    self.cancel_order(orderid)
            # 持有仓位
            else:
                if not self.active_limit_orders:
                    close_pos, D_KliqPoint = self.get_liqka_close_intense(
                        bar.symbol)
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
            # 持有空仓

    def on_cancel(self, order: OrderData):
        """
        Callback of cancelled order
        """
        # 平仓撤单，重新平仓
        if order.offset == Offset.CLOSE:
            self.send_order(vt_symbol=order.vt_symbol, direction=order.direction,
                            offset=order.offset, price=int(self.reclose_price), volume=1)

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """

        if trade.offset == Offset.OPEN:
            if trade.direction == Direction.LONG:
                self.lowafterentry = trade.price
            else:
                self.highafterentry = trade.price
