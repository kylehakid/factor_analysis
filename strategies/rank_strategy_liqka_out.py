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

        self.load_bar(days=60)
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

        rank = pd.qcut(
            self.am.close, q=self.bins, labels=False, duplicates="drop")[-1]
        # print(rank)
        if rank == 0:
            open_intense = -1
        self.update_intense(dt=bar.datetime, open_intense=open_intense)

    def on_1h_bar(self, bar: BarData):
        return

    def on_xh_bar(self, bar: BarData):
        return

# author = SJX


class Test01(Template):
    def __init__(self, engine, strategy_name: str,  setting: dict):
        # 参数
        super().__init__(engine, strategy_name, setting)
        setting.values()
        self.signal1: Signal = self.add_signal(
            TestDL, "test",  self.vt_symbols, self.test)

        self._TRS = 0.03
        self._delta_t = 0.003
        self._min_thre = 0.5

        self.risky_window = 10
        self.highafterentry = 0
        self.lowafterentry = 0
        self.reclose_price = 0
        self.liQKA = 0
        self.DliqPoint = 0
        self.KliqPoint = 0
        self.long_dict: dict = {}
        self.short_dict: dict = {}
        self.entry_short_ids: int = 0
        self.entry_long_ids: int = 0
        self.long_entry_price = np.empty(1000000)
        self.short_entry_price = np.empty(1000000)
        self.long_exit_price = np.empty(1000000)
        self.short_exit_price = np.empty(1000000)
        self.long_exit_count = 0
        self.short_exit_count = 0
        self.long_rtn_trades = 1
        self.short_rtn_trades = 1

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

        if self.inited():
            # 没有仓位
            # if self.pos > 5 or self.pos < -5:
            #     raise Exception(self.pos)
            open_intense = self.get_open_intense()
            if -30 <= self.pos <= 30:  # 限制持仓数量
                if not self.get_at_risk():
                    if open_intense > 0:
                        long_signal = True

                    elif open_intense < 0:
                        short_signal = True

                    if not self.active_limit_orders:
                        if (long_signal is True) and (self.long_rtn_trades > 0):
                            self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.LONG,
                                            offset=Offset.OPEN, price=bar.close,
                                            volume=1)

                        elif (short_signal is True) and (self.short_rtn_trades > 0):
                            self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.SHORT,
                                            offset=Offset.OPEN, price=bar.close,
                                            volume=1)
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
                                        if long_signal:
                                            self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.LONG,
                                                            offset=Offset.OPEN, price=bar.close * 1.01,
                                                            volume=1)
                                # 空开撤单
                                else:
                                    if order.price - bar.close > 5:
                                        # self.log_info(
                                        #     f'空开撤单,price:{order.price} - low:{bar.low} > 5')
                                        self.cancel_order(orderid)
                                        # 撤单后再开仓
                                        if short_signal:
                                            self.send_order(vt_symbol=bar.vt_symbol, direction=Direction.SHORT,
                                                            offset=Offset.OPEN, price=bar.close*0.99,
                                                            volume=1)

            # 有仓位,平仓逻辑
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

            # 记录出入场价格, 计算信号触发的收益
            if long_signal:
                # 记录入场价格
                self.long_entry_price[self.entry_long_ids] = bar.close
                self.long_dict[self.entry_long_ids] = {
                    "symbol": bar.vt_symbol,
                    "entry_price": bar.close,
                    "datetime": bar.datetime,
                    "highafterentry": bar.low,
                    "delta_t": 0,
                    "count": 0,
                    "stop_loss": bar.low*(1-self._TRS)}
                self.entry_long_ids += 1

                for ids, entry_dict in self.long_dict.copy().items():
                    self.long_dict[ids]["highafterentry"] = max(
                        entry_dict["highafterentry"], bar.low)  # 记录最高价

                    self.long_dict[ids]["count"] += 1  # 记录最高价持续时间
                    liqka = max(1.0 - self._delta_t *
                                self.long_dict[ids]["count"], self._min_thre)
                    self.long_dict[ids]["stop_loss"] = self.long_dict[ids]["highafterentry"] * (
                        1.0 - self._TRS * liqka)

                    if bar.low < self.long_dict[ids]["stop_loss"]:
                        self.long_exit_price[ids] = bar.close
                        del self.long_dict[ids]
                        self.long_exit_count += 1

            if short_signal:
                # 记录入场价格
                self.short_entry_price[self.entry_short_ids] = bar.close
                self.short_dict[self.entry_short_ids] = {
                    "symbol": bar.vt_symbol,
                    "entry_price": bar.close,
                    "datetime": bar.datetime,
                    "lowafterentry": bar.high,
                    "delta_t": 0,
                    "count": 0,
                    "stop_loss": bar.high*(1+self._TRS)}
                self.entry_short_ids += 1

                for ids, entry_dict in self.short_dict.copy().items():
                    self.short_dict[ids]["lowafterentry"] = min(
                        entry_dict["lowafterentry"], bar.high)  # 记录最低价

                    self.short_dict[ids]["count"] += 1  # 记录最低价持续时间
                    liqka = max(1.0 - self._delta_t *
                                self.short_dict[ids]["count"], self._min_thre)
                    self.short_dict[ids]["stop_loss"] = self.short_dict[ids]["lowafterentry"] * (
                        1.0 + self._TRS * liqka)

                    if bar.high > self.short_dict[ids]["stop_loss"]:
                        self.short_exit_price[ids] = bar.close
                        del self.short_dict[ids]
                        self.short_exit_count += 1
                        # print(self.short_exit_count)

            if self.long_exit_count > self.risky_window:
                self.long_rtn_trades = sum(self.long_exit_price[self.long_exit_count -
                                                                self.risky_window:self.long_exit_count] -
                                           self.long_entry_price[self.long_exit_count -
                                                                 self.risky_window:self.long_exit_count])
            if self.short_exit_count > self.risky_window:
                self.short_rtn_trades = sum(self.short_entry_price[self.short_exit_count -
                                                                   self.risky_window:self.short_exit_count] -
                                            self.short_exit_price[self.short_exit_count -
                                                                  self.risky_window:self.short_exit_count])
            # print(
            #     f"short_exit_count):{self.short_exit_count}, short_rtn_trades:{self.short_rtn_trades}")

    def on_cancel(self, order: OrderData):
        """
        Callback of cancelled order
        """
        pass

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        ...
