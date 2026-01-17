# A股策略示例库

## 核心原则

**不要试图记住所有策略，而是学会如何开发策略。**

本文档提供：
1. 策略开发方法论
2. 4种策略模板（技术指标、基本面、筹码分析、组合策略）
3. 10-15个典型策略示例
4. 如何开发新策略的指导

---

## 策略开发方法论

### 策略开发流程

```
1. 确定策略类型
   ↓
2. 查询相关指标（使用 Context7）
   ↓
3. 获取数据（参考 data-reference.md）
   ↓
4. 计算指标和生成信号
   ↓
5. 回测验证
   ↓
6. 调试优化（参考 debugging-guide.md）
```

### 如何使用 Context7 查询指标

在开发策略时，遇到不确定的指标计算方法，必须使用 Context7 查询：

```
技术指标查询：
Library ID: /websites/backtrader_docu
查询示例：
- "MACD indicator parameters fastperiod slowperiod"
- "RSI indicator calculation period"
- "Bollinger Bands indicator nbdevup nbdevdn"
- "SMA EMA crossover strategy"

财务指标查询：
Library ID: /websites/tushare_pro_document
查询示例：
- "fina_indicator roe eps 财务指标"
- "income revenue profit 利润表"
- "balancesheet total_assets 资产负债表"
- "daily_basic pe pb 估值指标"
```

### 策略验证方法

开发完策略后，必须进行分层验证：

1. **技术层验证**：代码能运行，无语法错误
2. **数据层验证**：数据格式正确，无缺失值或异常值
3. **逻辑层验证**：买卖信号符合预期
4. **结果层验证**：回测结果合理，收益率、夏普比率、最大回撤在合理范围

```python
# 验证代码模板
print(f"数据形状: {df.shape}")
print(f"数据范围: {df.index[0]} 到 {df.index[-1]}")
print(f"缺失值: {df.isnull().sum().sum()}")
print(f"买入信号数量: {buy_signals.sum()}")
print(f"卖出信号数量: {sell_signals.sum()}")
```

---

## 策略模板

### 技术指标策略模板

技术指标策略基于价格和成交量的历史数据，通过计算技术指标生成买卖信号。

```python
import tushare as ts
import backtrader as bt
import pandas as pd

# ========== 1. 初始化 Tushare ==========
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# ========== 2. 定义策略 ==========
class TechnicalStrategy(bt.Strategy):
    """技术指标策略模板"""
    params = (
        ('indicator_period', 20),  # 指标周期参数
    )

    def __init__(self):
        # 计算技术指标（示例：移动平均线）
        self.indicator = bt.indicators.SMA(self.data.close, period=self.p.indicator_period)

        # 记录买入bar（用于T+1规则）
        self.buy_bar = None

    def next(self):
        # T+1规则：检查是否可以卖出
        if self.position:
            if self.buy_bar is not None:
                bars_held = len(self) - self.buy_bar
                if bars_held >= 1:  # 持有超过1个交易日
                    # 卖出条件（根据具体策略修改）
                    if self.data.close[0] < self.indicator[0]:
                        self.sell()
                        self.buy_bar = None
        else:
            # 买入条件（根据具体策略修改）
            if self.data.close[0] > self.indicator[0]:
                self.buy()
                self.buy_bar = len(self)

# ========== 3. 获取数据 ==========
df = pro.daily(ts_code='600000.SH', start_date='20220101', end_date='20231231')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()
df = df.rename(columns={'vol': 'volume'})
df = df[['open', 'high', 'low', 'close', 'volume']]

# ========== 4. 运行回测 ==========
cerebro = bt.Cerebro()
cerebro.addstrategy(TechnicalStrategy)
cerebro.adddata(bt.feeds.PandasData(dataname=df))
cerebro.broker.setcash(100000)

# 设置A股手续费
class AShareCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.0003),
        ('stamp_duty', 0.001),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),
    )
    def _getcommission(self, size, price, pseudoexec):
        turnover = abs(size) * price
        commission = max(turnover * self.p.commission, 5)
        if size < 0:
            commission += turnover * self.p.stamp_duty
        return commission

cerebro.broker.addcommissioninfo(AShareCommission())

# 添加分析器
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

print(f'初始资金: {cerebro.broker.getvalue():.2f}')
results = cerebro.run()
print(f'最终资金: {cerebro.broker.getvalue():.2f}')

# 打印分析结果
strat = results[0]
print(f"年化收益: {strat.analyzers.returns.get_analysis().get('rnorm100', 0):.2f}%")
print(f"夏普比率: {strat.analyzers.sharpe.get_analysis().get('sharperatio', 0):.2f}")
print(f"最大回撤: {strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0):.2f}%")
```

**模板使用说明**：
1. 修改 `indicator` 部分，计算你需要的技术指标
2. 修改买入条件和卖出条件
3. 调整参数（如周期、阈值等）
4. 运行回测并验证结果

---

### 基本面策略模板

基本面策略基于公司财务数据和估值指标，选择基本面优秀的股票。

```python
import tushare as ts
import backtrader as bt
import pandas as pd
import numpy as np

# ========== 1. 初始化 Tushare ==========
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# ========== 2. 获取股票池 ==========
# 获取所有A股列表（过滤ST股票）
stock_list = pro.stock_basic(list_status='L', fields='ts_code,name')
stock_list = stock_list[~stock_list['name'].str.contains('ST')]
stock_codes = stock_list['ts_code'].tolist()[:50]  # 示例：取前50只股票

# ========== 3. 获取财务数据 ==========
def get_fundamental_data(ts_code, start_date, end_date):
    """获取财务指标数据"""
    # 获取财务指标（ROE、PE、PB等）
    df_indicator = pro.fina_indicator(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields='ts_code,end_date,roe,eps,bps'
    )

    if df_indicator.empty:
        return None

    # 转换日期格式
    df_indicator['end_date'] = pd.to_datetime(df_indicator['end_date'])
    df_indicator = df_indicator.set_index('end_date').sort_index()

    return df_indicator

# ========== 4. 定义策略 ==========
class FundamentalStrategy(bt.Strategy):
    """基本面策略模板"""
    params = (
        ('roe_threshold', 10),  # ROE阈值（%）
        ('rebalance_days', 90),  # 调仓周期（天）
    )

    def __init__(self):
        self.order = None
        self.bar_executed = 0

    def next(self):
        # 定期调仓
        if len(self) % self.p.rebalance_days == 0:
            # 这里应该根据最新财务数据筛选股票
            # 简化示例：保持持仓
            pass

# ========== 5. 基本面选股函数 ==========
def select_stocks_by_fundamental(stock_codes, trade_date, pro):
    """
    根据基本面指标选股

    参数:
        stock_codes: 股票代码列表
        trade_date: 交易日期
        pro: tushare接口

    返回:
        选中的股票代码列表
    """
    selected_stocks = []

    for ts_code in stock_codes:
        # 获取最新财务指标
        df = pro.fina_indicator(
            ts_code=ts_code,
            end_date=trade_date,
            fields='ts_code,end_date,roe,eps'
        )

        if df.empty:
            continue

        # 筛选条件：ROE > 10%
        latest_roe = df.iloc[0]['roe']
        if latest_roe > 10:
            selected_stocks.append(ts_code)

    return selected_stocks

# 使用示例
selected = select_stocks_by_fundamental(stock_codes[:10], '20231231', pro)
print(f"选中股票: {selected}")
```

**模板使用说明**：
1. 修改 `get_fundamental_data` 函数，获取你需要的财务指标
2. 修改 `select_stocks_by_fundamental` 函数中的筛选条件
3. 实现定期调仓逻辑
4. 注意财务数据的发布时间，避免前视偏差

---

### 筹码分析策略模板

筹码分析策略基于资金流向、机构持仓等数据，跟踪主力资金动向。

```python
import tushare as ts
import backtrader as bt
import pandas as pd

# ========== 1. 初始化 Tushare ==========
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# ========== 2. 获取筹码数据 ==========
def get_moneyflow_data(ts_code, start_date, end_date):
    """获取资金流向数据"""
    df = pro.moneyflow(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )

    if df.empty:
        return None

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()

    # 计算大单净流入
    df['big_net_inflow'] = (df['buy_lg_amount'] + df['buy_elg_amount']) - \
                           (df['sell_lg_amount'] + df['sell_elg_amount'])

    return df

# ========== 3. 定义策略 ==========
class MoneyFlowStrategy(bt.Strategy):
    """资金流向策略模板"""
    params = (
        ('flow_period', 5),  # 资金流向统计周期
        ('flow_threshold', 0),  # 资金流向阈值
    )

    def __init__(self):
        # 这里需要将资金流向数据与价格数据对齐
        # 简化示例：使用价格数据
        self.buy_bar = None

    def next(self):
        # T+1规则
        if self.position:
            if self.buy_bar is not None:
                bars_held = len(self) - self.buy_bar
                if bars_held >= 1:
                    # 卖出条件：根据资金流向判断
                    # 这里需要访问资金流向数据
                    self.sell()
                    self.buy_bar = None
        else:
            # 买入条件：大单持续净流入
            # 这里需要访问资金流向数据
            self.buy()
            self.buy_bar = len(self)

# 使用示例
df_flow = get_moneyflow_data('600000.SH', '20230101', '20231231')
if df_flow is not None:
    print(f"资金流向数据:\n{df_flow.head()}")
    print(f"大单净流入均值: {df_flow['big_net_inflow'].mean():.2f}万元")
```

**模板使用说明**：
1. 选择筹码数据类型（资金流向、融资融券、北向资金等）
2. 计算筹码指标（如大单净流入、主力持仓变化等）
3. 将筹码数据与价格数据对齐
4. 根据筹码指标生成买卖信号

---

### 组合策略模板

组合策略结合多个因子，通过多维度筛选提高策略稳定性。

```python
import tushare as ts
import backtrader as bt
import pandas as pd

# ========== 1. 初始化 Tushare ==========
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# ========== 2. 定义多因子策略 ==========
class MultiFactorStrategy(bt.Strategy):
    """多因子组合策略模板"""
    params = (
        ('sma_period', 20),      # 技术指标：均线周期
        ('rsi_period', 14),      # 技术指标：RSI周期
        ('rsi_oversold', 30),    # RSI超卖阈值
        ('rsi_overbought', 70),  # RSI超买阈值
    )

    def __init__(self):
        # 技术指标
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.sma_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

        # T+1规则
        self.buy_bar = None

    def next(self):
        # T+1规则
        if self.position:
            if self.buy_bar is not None:
                bars_held = len(self) - self.buy_bar
                if bars_held >= 1:
                    # 卖出条件：多个条件组合
                    # 条件1：价格跌破均线
                    cond1 = self.data.close[0] < self.sma[0]
                    # 条件2：RSI超买
                    cond2 = self.rsi[0] > self.p.rsi_overbought

                    if cond1 or cond2:
                        self.sell()
                        self.buy_bar = None
        else:
            # 买入条件：多个条件组合
            # 条件1：价格突破均线
            cond1 = self.data.close[0] > self.sma[0]
            # 条件2：RSI超卖反弹
            cond2 = self.rsi[0] > self.p.rsi_oversold and self.rsi[-1] < self.p.rsi_oversold

            if cond1 and cond2:
                self.buy()
                self.buy_bar = len(self)

# ========== 3. 运行回测 ==========
# （数据获取和回测代码与技术指标策略模板相同）
```

**模板使用说明**：
1. 选择要组合的因子（技术指标、基本面、筹码等）
2. 定义每个因子的计算方法和阈值
3. 设计因子组合逻辑（AND、OR、加权等）
4. 测试不同因子组合的效果

---

## 典型策略示例

以下提供10-15个典型策略示例，涵盖技术指标、基本面、筹码分析和组合策略四大类。

---

### 技术指标类策略

#### 示例1：MACD金叉策略

MACD（Moving Average Convergence Divergence）是常用的趋势跟踪指标，当MACD线上穿信号线时产生买入信号。

```python
import tushare as ts
import backtrader as bt
import pandas as pd

# 初始化 Tushare
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 定义策略
class MACDStrategy(bt.Strategy):
    """MACD金叉策略"""
    params = (
        ('fast_period', 12),   # 快线周期
        ('slow_period', 26),   # 慢线周期
        ('signal_period', 9),  # 信号线周期
    )

    def __init__(self):
        # 计算MACD指标
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.fast_period,
            period_me2=self.p.slow_period,
            period_signal=self.p.signal_period
        )

        # T+1规则：记录买入bar
        self.buy_bar = None

    def next(self):
        # T+1规则：检查是否可以卖出
        if self.position:
            if self.buy_bar is not None:
                bars_held = len(self) - self.buy_bar
                if bars_held >= 1:
                    # 卖出条件：MACD死叉
                    if self.macd.macd[0] < self.macd.signal[0] and \
                       self.macd.macd[-1] >= self.macd.signal[-1]:
                        self.sell()
                        self.buy_bar = None
        else:
            # 买入条件：MACD金叉
            if self.macd.macd[0] > self.macd.signal[0] and \
               self.macd.macd[-1] <= self.macd.signal[-1]:
                self.buy()
                self.buy_bar = len(self)

# 获取数据
df = pro.daily(ts_code='600519.SH', start_date='20220101', end_date='20231231')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()
df = df.rename(columns={'vol': 'volume'})
df = df[['open', 'high', 'low', 'close', 'volume']]

# 运行回测
cerebro = bt.Cerebro()
cerebro.addstrategy(MACDStrategy)
cerebro.adddata(bt.feeds.PandasData(dataname=df))
cerebro.broker.setcash(100000)

# 设置A股手续费
class AShareCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.0003),
        ('stamp_duty', 0.001),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),
    )
    def _getcommission(self, size, price, pseudoexec):
        turnover = abs(size) * price
        commission = max(turnover * self.p.commission, 5)
        if size < 0:
            commission += turnover * self.p.stamp_duty
        return commission

cerebro.broker.addcommissioninfo(AShareCommission())

# 添加分析器
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

print(f'初始资金: {cerebro.broker.getvalue():.2f}')
results = cerebro.run()
print(f'最终资金: {cerebro.broker.getvalue():.2f}')

# 打印分析结果
strat = results[0]
print(f"年化收益: {strat.analyzers.returns.get_analysis().get('rnorm100', 0):.2f}%")
print(f"夏普比率: {strat.analyzers.sharpe.get_analysis().get('sharperatio', 0):.2f}")
print(f"最大回撤: {strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0):.2f}%")
```

**策略说明**：
- **买入信号**：MACD线上穿信号线（金叉）
- **卖出信号**：MACD线下穿信号线（死叉）
- **适用场景**：趋势明显的市场
- **注意事项**：震荡市场容易产生虚假信号

---

#### 示例2：RSI超卖反弹策略

RSI（Relative Strength Index）是衡量超买超卖的指标，当RSI低于30时认为超卖，可能出现反弹。

```python
import tushare as ts
import backtrader as bt
import pandas as pd

# 初始化 Tushare
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 定义策略
class RSIOversoldStrategy(bt.Strategy):
    """RSI超卖反弹策略"""
    params = (
        ('rsi_period', 14),      # RSI周期
        ('oversold', 30),        # 超卖阈值
        ('overbought', 70),      # 超买阈值
    )

    def __init__(self):
        # 计算RSI指标
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

        # T+1规则
        self.buy_bar = None

    def next(self):
        # T+1规则
        if self.position:
            if self.buy_bar is not None:
                bars_held = len(self) - self.buy_bar
                if bars_held >= 1:
                    # 卖出条件：RSI超买或跌破买入价10%
                    if self.rsi[0] > self.p.overbought:
                        self.sell()
                        self.buy_bar = None
        else:
            # 买入条件：RSI从超卖区反弹
            if self.rsi[0] > self.p.oversold and self.rsi[-1] <= self.p.oversold:
                self.buy()
                self.buy_bar = len(self)

# 获取数据
df = pro.daily(ts_code='000001.SZ', start_date='20220101', end_date='20231231')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()
df = df.rename(columns={'vol': 'volume'})
df = df[['open', 'high', 'low', 'close', 'volume']]

# 运行回测（代码与示例1相同）
cerebro = bt.Cerebro()
cerebro.addstrategy(RSIOversoldStrategy)
cerebro.adddata(bt.feeds.PandasData(dataname=df))
cerebro.broker.setcash(100000)

# 设置手续费（代码与示例1相同）
class AShareCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.0003),
        ('stamp_duty', 0.001),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),
    )
    def _getcommission(self, size, price, pseudoexec):
        turnover = abs(size) * price
        commission = max(turnover * self.p.commission, 5)
        if size < 0:
            commission += turnover * self.p.stamp_duty
        return commission

cerebro.broker.addcommissioninfo(AShareCommission())
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

print(f'初始资金: {cerebro.broker.getvalue():.2f}')
results = cerebro.run()
print(f'最终资金: {cerebro.broker.getvalue():.2f}')

strat = results[0]
print(f"年化收益: {strat.analyzers.returns.get_analysis().get('rnorm100', 0):.2f}%")
print(f"夏普比率: {strat.analyzers.sharpe.get_analysis().get('sharperatio', 0):.2f}")
print(f"最大回撤: {strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0):.2f}%")
```

**策略说明**：
- **买入信号**：RSI从超卖区（<30）反弹
- **卖出信号**：RSI进入超买区（>70）
- **适用场景**：震荡市场，寻找短期反弹机会
- **注意事项**：下跌趋势中可能持续超卖

---

#### 示例3：布林带突破策略

布林带（Bollinger Bands）由中轨（移动平均线）和上下轨（标准差）组成，价格突破上轨可能预示强势。

```python
import tushare as ts
import backtrader as bt
import pandas as pd

# 初始化 Tushare
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 定义策略
class BollingerBreakoutStrategy(bt.Strategy):
    """布林带突破策略"""
    params = (
        ('period', 20),      # 布林带周期
        ('devfactor', 2.0),  # 标准差倍数
    )

    def __init__(self):
        # 计算布林带指标
        self.boll = bt.indicators.BollingerBands(
            self.data.close,
            period=self.p.period,
            devfactor=self.p.devfactor
        )

        # T+1规则
        self.buy_bar = None

    def next(self):
        # T+1规则
        if self.position:
            if self.buy_bar is not None:
                bars_held = len(self) - self.buy_bar
                if bars_held >= 1:
                    # 卖出条件：价格跌破中轨
                    if self.data.close[0] < self.boll.mid[0]:
                        self.sell()
                        self.buy_bar = None
        else:
            # 买入条件：价格突破上轨
            if self.data.close[0] > self.boll.top[0] and \
               self.data.close[-1] <= self.boll.top[-1]:
                self.buy()
                self.buy_bar = len(self)

# 获取数据
df = pro.daily(ts_code='600036.SH', start_date='20220101', end_date='20231231')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()
df = df.rename(columns={'vol': 'volume'})
df = df[['open', 'high', 'low', 'close', 'volume']]

# 运行回测
cerebro = bt.Cerebro()
cerebro.addstrategy(BollingerBreakoutStrategy)
cerebro.adddata(bt.feeds.PandasData(dataname=df))
cerebro.broker.setcash(100000)

# 设置手续费
class AShareCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.0003),
        ('stamp_duty', 0.001),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),
    )
    def _getcommission(self, size, price, pseudoexec):
        turnover = abs(size) * price
        commission = max(turnover * self.p.commission, 5)
        if size < 0:
            commission += turnover * self.p.stamp_duty
        return commission

cerebro.broker.addcommissioninfo(AShareCommission())
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

print(f'初始资金: {cerebro.broker.getvalue():.2f}')
results = cerebro.run()
print(f'最终资金: {cerebro.broker.getvalue():.2f}')

strat = results[0]
print(f"年化收益: {strat.analyzers.returns.get_analysis().get('rnorm100', 0):.2f}%")
print(f"夏普比率: {strat.analyzers.sharpe.get_analysis().get('sharperatio', 0):.2f}")
print(f"最大回撤: {strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0):.2f}%")
```

**策略说明**：
- **买入信号**：价格突破布林带上轨
- **卖出信号**：价格跌破布林带中轨
- **适用场景**：趋势启动阶段
- **注意事项**：假突破风险较高，建议结合成交量确认

---

#### 示例4：双均线交叉策略

双均线策略是最经典的趋势跟踪策略，通过短期均线和长期均线的交叉产生信号。

```python
import tushare as ts
import backtrader as bt
import pandas as pd

# 初始化 Tushare
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 定义策略
class DualSMAStrategy(bt.Strategy):
    """双均线交叉策略"""
    params = (
        ('fast_period', 5),   # 快速均线周期
        ('slow_period', 20),  # 慢速均线周期
    )

    def __init__(self):
        # 计算快慢均线
        self.fast_sma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)

        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

        # T+1规则
        self.buy_bar = None

    def next(self):
        # T+1规则
        if self.position:
            if self.buy_bar is not None:
                bars_held = len(self) - self.buy_bar
                if bars_held >= 1:
                    # 卖出条件：死叉
                    if self.crossover[0] < 0:
                        self.sell()
                        self.buy_bar = None
        else:
            # 买入条件：金叉
            if self.crossover[0] > 0:
                self.buy()
                self.buy_bar = len(self)

# 获取数据
df = pro.daily(ts_code='600000.SH', start_date='20220101', end_date='20231231')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()
df = df.rename(columns={'vol': 'volume'})
df = df[['open', 'high', 'low', 'close', 'volume']]

# 运行回测
cerebro = bt.Cerebro()
cerebro.addstrategy(DualSMAStrategy)
cerebro.adddata(bt.feeds.PandasData(dataname=df))
cerebro.broker.setcash(100000)

# 设置手续费
class AShareCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.0003),
        ('stamp_duty', 0.001),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),
    )
    def _getcommission(self, size, price, pseudoexec):
        turnover = abs(size) * price
        commission = max(turnover * self.p.commission, 5)
        if size < 0:
            commission += turnover * self.p.stamp_duty
        return commission

cerebro.broker.addcommissioninfo(AShareCommission())
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

print(f'初始资金: {cerebro.broker.getvalue():.2f}')
results = cerebro.run()
print(f'最终资金: {cerebro.broker.getvalue():.2f}')

strat = results[0]
print(f"年化收益: {strat.analyzers.returns.get_analysis().get('rnorm100', 0):.2f}%")
print(f"夏普比率: {strat.analyzers.sharpe.get_analysis().get('sharperatio', 0):.2f}")
print(f"最大回撤: {strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0):.2f}%")
```

**策略说明**：
- **买入信号**：快速均线上穿慢速均线（金叉）
- **卖出信号**：快速均线下穿慢速均线（死叉）
- **适用场景**：趋势明显的市场
- **注意事项**：震荡市场会频繁交易，产生较多手续费

---

### 基本面类策略

#### 示例5：低市盈率策略

选择市盈率较低的股票，适合价值投资。

**策略思路**：
1. 获取所有A股的市盈率数据
2. 过滤掉ST股票和市盈率为负的股票
3. 选择市盈率最低的前20只股票
4. 定期调仓（如每季度）

**关键代码片段**：
```python
# 获取市盈率数据
df_basic = pro.daily_basic(trade_date='20231231', fields='ts_code,pe')
df_basic = df_basic[df_basic['pe'] > 0]  # 过滤负市盈率
df_basic = df_basic.sort_values('pe').head(20)  # 选择最低的20只
```

**注意事项**：
- 需要结合其他指标（如ROE、负债率）避免价值陷阱
- 定期调仓，避免持有基本面恶化的股票

---

#### 示例6：高ROE策略

选择净资产收益率（ROE）高的优质公司。

**策略思路**：
1. 获取财务指标数据
2. 筛选ROE > 15%的股票
3. 结合营收增长率进一步筛选
4. 季度调仓

**关键代码片段**：
```python
# 获取财务指标
df_indicator = pro.fina_indicator(
    period='20231231',
    fields='ts_code,roe,revenue_yoy'
)
# 筛选条件
selected = df_indicator[(df_indicator['roe'] > 15) &
                        (df_indicator['revenue_yoy'] > 10)]
```

**注意事项**：
- 注意财务数据的发布时间，避免前视偏差
- ROE过高可能存在财务造假风险，需要交叉验证

---

#### 示例7：营收加速策略

选择营收增速加快的成长股。

**策略思路**：
1. 获取月度营收数据
2. 计算3个月平均营收同比增速
3. 选择增速最快的股票
4. 月度调仓

**关键代码片段**：
```python
# 获取营收数据（需要查询Context7确认接口）
# Library ID: /websites/tushare_pro_document
# 查询: "income revenue 营收 利润表"

# 计算营收增速
revenue_growth = (revenue.rolling(3).mean().pct_change(12) * 100)
selected = revenue_growth.nlargest(20)
```

**注意事项**：
- 营收数据按月发布，需要对齐到交易日
- 注意季节性因素的影响

---

### 筹码分析类策略

#### 示例8：北向资金流入策略

跟踪北向资金（外资通过沪深股通买入A股）的流向。

**策略思路**：
1. 获取沪深股通资金流向数据
2. 计算连续N天净流入的股票
3. 选择净流入金额最大的股票

**关键代码片段**：
```python
# 获取沪深股通数据（需要查询Context7）
# Library ID: /websites/tushare_pro_document
# 查询: "hsgt_top10 沪深股通 北向资金"

# 计算净流入
net_inflow = buy_amount - sell_amount
continuous_inflow = (net_inflow > 0).rolling(5).sum() >= 5
```

**注意事项**：
- 北向资金数据可能有延迟
- 需要结合市场整体情况判断

---

#### 示例9：融资融券策略

利用融资融券数据判断市场情绪。

**策略思路**：
1. 获取融资融券余额数据
2. 计算融资买入占比
3. 融资买入占比上升表示看多情绪增强

**关键代码片段**：
```python
# 获取融资融券数据
df_margin = pro.margin(ts_code='600000.SH',
                       start_date='20230101',
                       end_date='20231231')
# 计算融资占比
margin_ratio = df_margin['rzye'] / (df_margin['rzye'] + df_margin['rqye'])
```

**注意事项**：
- 融资余额过高可能预示风险
- 需要结合个股基本面判断

---

### 组合策略类

#### 示例10：技术面+基本面组合策略

结合技术指标和基本面指标，提高策略稳定性。

**策略思路**：
1. 基本面筛选：ROE > 10%，PE < 30
2. 技术面筛选：价格突破20日均线
3. 两个条件同时满足时买入

**关键代码片段**：
```python
class CombinedStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=20)
        # 基本面数据需要预先加载并对齐

    def next(self):
        # 技术面条件
        tech_signal = self.data.close[0] > self.sma[0]
        # 基本面条件（需要从外部数据获取）
        # fund_signal = (roe > 10) and (pe < 30)

        if tech_signal:  # 简化示例
            if not self.position:
                self.buy()
```

**注意事项**：
- 需要解决技术数据和基本面数据的对齐问题
- 基本面数据更新频率低，需要前向填充

---

#### 示例11：多因子轮动策略

根据多个因子的综合得分选股，定期轮动。

**策略思路**：
1. 计算多个因子（动量、价值、质量）
2. 对每个因子进行标准化和打分
3. 综合得分最高的股票入选
4. 月度或季度调仓

**关键代码片段**：
```python
# 计算动量因子
momentum = (close / close.shift(20) - 1)

# 计算价值因子（低PE）
value = 1 / pe

# 计算质量因子（高ROE）
quality = roe

# 标准化并综合
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
momentum_norm = scaler.fit_transform(momentum.values.reshape(-1, 1))
value_norm = scaler.fit_transform(value.values.reshape(-1, 1))
quality_norm = scaler.fit_transform(quality.values.reshape(-1, 1))

# 综合得分（可以设置权重）
composite_score = 0.4 * momentum_norm + 0.3 * value_norm + 0.3 * quality_norm
```

**注意事项**：
- 因子之间可能存在相关性，需要去相关处理
- 权重设置需要通过回测优化

---

## 如何开发新策略

### 步骤1：确定策略类型

根据你的投资理念选择策略类型：
- **技术指标策略**：适合短期交易，关注价格和成交量
- **基本面策略**：适合中长期投资，关注公司质量
- **筹码分析策略**：跟踪主力资金，捕捉市场情绪
- **组合策略**：结合多个维度，提高稳定性

### 步骤2：查询相关指标

使用 Context7 查询你需要的指标计算方法：

```
技术指标：
Library ID: /websites/backtrader_docu
查询示例："KDJ indicator stochastic oscillator"

财务指标：
Library ID: /websites/tushare_pro_document
查询示例："fina_indicator 财务指标 roe eps"
```

### 步骤3：参考对应模板

根据策略类型，参考本文档中的对应模板：
- 技术指标策略 → 技术指标策略模板
- 基本面策略 → 基本面策略模板
- 筹码分析策略 → 筹码分析策略模板
- 组合策略 → 组合策略模板

### 步骤4：开发和测试

1. **编写策略代码**：基于模板修改买卖条件
2. **数据验证**：检查数据格式、缺失值、异常值
3. **逻辑验证**：打印买卖信号，确认符合预期
4. **回测验证**：运行回测，查看收益率、夏普比率、最大回撤

### 步骤5：调试优化

参考 [debugging-guide.md](debugging-guide.md) 进行调试：
- 使用分层验证方法
- 检查T+1规则是否正确实现
- 检查手续费设置是否正确
- 检查是否存在前视偏差

---

## 最佳实践

1. **遵循A股特殊规则**：T+1、涨跌停、ST股过滤、手续费设置
2. **避免前视偏差**：不要使用未来数据，注意财务数据发布时间
3. **控制交易频率**：频繁交易会产生大量手续费
4. **分散投资**：不要把所有资金投入单只股票
5. **设置止损**：控制单笔交易的最大亏损
6. **定期调仓**：基本面策略建议季度调仓，技术策略可以更频繁
7. **回测验证**：任何策略上线前都要经过充分回测

---

## 相关文档

- [data-reference.md](data-reference.md) - 数据获取指南
- [dataframe-reference.md](dataframe-reference.md) - 数据处理工具
- [ashare-rules.md](ashare-rules.md) - A股特殊规则
- [debugging-guide.md](debugging-guide.md) - 调试指南
- [best-practices.md](best-practices.md) - 最佳实践

---

## 总结

**核心原则**：
1. ✅ 不要试图记住所有策略，学会开发方法论
2. ✅ 遇到不确定的指标，使用 Context7 查询
3. ✅ 参考模板和典型示例，举一反三
4. ✅ 遵循分层验证，确保策略正确性
5. ✅ 遵守A股特殊规则，避免常见错误

**下一步**：
- 选择一个策略类型开始实践
- 查询相关指标的计算方法
- 基于模板开发你的第一个策略
- 充分回测和调试后再实盘
