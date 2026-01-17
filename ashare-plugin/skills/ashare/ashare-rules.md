# A股特殊交易规则

本文档说明 A 股市场的特殊交易规则，以及如何在 Backtrader 回测中正确处理这些规则。

---

## T+1 交易规则

A股实行 T+1 交易制度：当日买入的股票，次日才能卖出。

### 实现方式

```python
import backtrader as bt

class T1Strategy(bt.Strategy):
    """支持 T+1 规则的策略基类"""

    def __init__(self):
        # 记录每只股票的买入 bar 序号
        self.buy_bars = {}

    def next(self):
        for i, d in enumerate(self.datas):
            pos = self.getposition(d)

            if pos.size > 0:
                # 有持仓，检查是否满足 T+1
                if d._name in self.buy_bars:
                    bars_held = len(self) - self.buy_bars[d._name]
                    if bars_held >= 1:
                        # 持有超过 1 个交易日，可以卖出
                        if self.should_sell(d):
                            self.sell(data=d)
                            del self.buy_bars[d._name]
            else:
                # 无持仓，检查是否买入
                if self.should_buy(d):
                    self.buy(data=d)
                    self.buy_bars[d._name] = len(self)

    def should_buy(self, data):
        """子类实现买入逻辑"""
        return False

    def should_sell(self, data):
        """子类实现卖出逻辑"""
        return False
```

### 使用示例

```python
class MySMAStrategy(T1Strategy):
    params = (('period', 20),)

    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.period)

    def should_buy(self, data):
        return data.close[0] > self.sma[0]

    def should_sell(self, data):
        return data.close[0] < self.sma[0]
```

---

## 涨跌停限制

A股有涨跌停板制度，不同板块限制不同：

| 板块 | 涨跌停幅度 | 代码特征 |
|------|-----------|---------|
| 主板 | ±10% | 60xxxx.SH, 00xxxx.SZ |
| 科创板 | ±20% | 688xxx.SH |
| 创业板 | ±20% | 30xxxx.SZ |
| ST/\*ST | ±5% | 名称含 ST |
| 北交所 | ±30% | 8xxxxx.BJ |

### 涨跌停判断函数

```python
def get_limit_pct(ts_code, name=''):
    """
    获取股票的涨跌停幅度

    参数:
        ts_code: 股票代码
        name: 股票名称（用于判断 ST）

    返回:
        涨跌停幅度（小数）
    """
    # ST 股票
    if 'ST' in name or '*ST' in name:
        return 0.05

    # 科创板
    if ts_code.startswith('688'):
        return 0.20

    # 创业板
    if ts_code.startswith('30'):
        return 0.20

    # 北交所
    if ts_code.startswith('8') and ts_code.endswith('.BJ'):
        return 0.30

    # 主板（默认）
    return 0.10


def check_price_limit(close, pre_close, ts_code, name=''):
    """
    检查是否涨跌停

    参数:
        close: 当前收盘价
        pre_close: 昨日收盘价
        ts_code: 股票代码
        name: 股票名称

    返回:
        (is_limit_up, is_limit_down): 是否涨停, 是否跌停
    """
    limit_pct = get_limit_pct(ts_code, name)

    # 涨跌停价格（四舍五入到分）
    up_limit = round(pre_close * (1 + limit_pct), 2)
    down_limit = round(pre_close * (1 - limit_pct), 2)

    is_limit_up = close >= up_limit
    is_limit_down = close <= down_limit

    return is_limit_up, is_limit_down
```

### 涨跌停对交易的影响

```python
class LimitAwareStrategy(bt.Strategy):
    """考虑涨跌停的策略"""

    def next(self):
        close = self.data.close[0]
        pre_close = self.data.close[-1] if len(self.data) > 1 else close

        # 简化判断（实际应根据股票代码判断）
        pct_change = (close - pre_close) / pre_close

        # 涨停：无法买入（封板）
        if pct_change >= 0.095:  # 接近10%涨停
            # 不执行买入
            return

        # 跌停：无法卖出（封板）
        if pct_change <= -0.095:  # 接近10%跌停
            # 不执行卖出
            return

        # 正常交易逻辑
        # ...
```

### 使用 Tushare 获取涨跌停价格

```python
def get_limit_prices(ts_code, trade_date, pro):
    """
    从 Tushare 获取涨跌停价格

    注意：需要查询 context7 确认 stk_limit 接口的具体用法
    Library ID: /websites/tushare_pro_document
    查询: "stk_limit 涨跌停价格"
    """
    df = pro.stk_limit(ts_code=ts_code, trade_date=trade_date)
    if not df.empty:
        return {
            'up_limit': df['up_limit'].iloc[0],
            'down_limit': df['down_limit'].iloc[0]
        }
    return None
```

---

## ST 股票处理

ST（Special Treatment）股票是被特别处理的股票，通常存在财务问题或其他风险。

### ST 股票特点

| 类型 | 说明 | 涨跌停 |
|------|------|--------|
| ST | 财务异常 | ±5% |
| \*ST | 退市风险警示 | ±5% |
| S | 未完成股改 | ±5% |
| SST | 未股改 + 财务异常 | ±5% |

### 过滤 ST 股票

```python
def filter_st_stocks(pro):
    """
    获取非 ST 股票列表

    返回:
        非 ST 股票的 ts_code 列表
    """
    # 获取所有上市股票
    df = pro.stock_basic(list_status='L', fields='ts_code,name')

    # 过滤掉 ST 相关股票
    st_pattern = r'ST|\*ST|S |SST'
    non_st = df[~df['name'].str.contains(st_pattern, regex=True)]

    return non_st['ts_code'].tolist()


def is_st_stock(name):
    """判断是否为 ST 股票"""
    st_keywords = ['ST', '*ST', 'S ', 'SST']
    return any(kw in name for kw in st_keywords)
```

### 在策略中排除 ST

```python
class NoSTStrategy(bt.Strategy):
    """排除 ST 股票的策略"""

    def __init__(self):
        self.st_stocks = set()  # 存储 ST 股票代码

    def prenext(self):
        # 在数据预热阶段更新 ST 列表
        pass

    def next(self):
        for d in self.datas:
            if d._name in self.st_stocks:
                continue  # 跳过 ST 股票

            # 正常交易逻辑
            # ...
```

---

## 手续费和印花税

A股交易费用包括：

| 费用类型 | 费率 | 收取方式 |
|---------|------|---------|
| 佣金 | 约万分之三 | 买卖双向 |
| 印花税 | 千分之一 | 仅卖出 |
| 过户费 | 万分之0.2 | 买卖双向（可忽略） |

### 自定义手续费类

```python
import backtrader as bt

class AShareCommission(bt.CommInfoBase):
    """
    A股手续费类

    - 佣金: 万分之三 (双向收取)
    - 印花税: 千分之一 (仅卖出收取)
    - 佣金最低 5 元
    """
    params = (
        ('commission', 0.0003),   # 佣金 万三
        ('stamp_duty', 0.001),    # 印花税 千一
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),  # 关键：commission 以小数形式传入
    )

    def _getcommission(self, size, price, pseudoexec):
        # 交易金额
        turnover = abs(size) * price

        # 佣金 (买卖都收)
        commission = turnover * self.p.commission

        # 佣金最低 5 元
        if commission < 5:
            commission = 5

        # 印花税 (仅卖出收取，size < 0 表示卖出)
        if size < 0:
            commission += turnover * self.p.stamp_duty

        return commission


# 使用方式
cerebro = bt.Cerebro()
cerebro.broker.addcommissioninfo(AShareCommission())
```

### 简化版（不考虑最低佣金）

```python
# 如果不需要精确计算，可以使用简化设置
cerebro.broker.setcommission(commission=0.001)  # 约等于佣金+印花税
```

---

## 交易单位限制

A股交易有最小单位限制：

| 规则 | 说明 |
|------|------|
| 买入 | 必须是 100 股（1手）的整数倍 |
| 卖出 | 可以卖出不足 100 股的零股 |

### 在 Backtrader 中实现

```python
class RoundLotSizer(bt.Sizer):
    """
    A股整手买入 Sizer

    确保买入数量是 100 的整数倍
    """
    params = (
        ('stake', 100),  # 每手股数
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            # 计算可买入的最大手数
            price = data.close[0]
            max_shares = int(cash / price)
            lots = max_shares // self.p.stake
            return lots * self.p.stake
        else:
            # 卖出时返回全部持仓
            position = self.broker.getposition(data)
            return position.size


# 使用方式
cerebro.addsizer(RoundLotSizer)
```

---

## 停牌处理

股票停牌期间无法交易。

### 检查停牌状态

```python
def get_suspend_info(ts_code, start_date, end_date, pro):
    """
    获取停牌信息

    注意：需要查询 context7 确认 suspend_d 接口的具体用法
    Library ID: /websites/tushare_pro_document
    查询: "suspend_d 停牌"
    """
    df = pro.suspend_d(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields='ts_code,trade_date,suspend_type'
    )
    return df
```

### 在回测中处理停牌

Backtrader 使用 PandasData 时，停牌日期的数据会自动缺失，不会产生交易信号。如果需要显式处理：

```python
class SuspendAwareStrategy(bt.Strategy):
    """考虑停牌的策略"""

    def next(self):
        # 检查今日是否有数据（停牌时无数据）
        if len(self.data) < 2:
            return

        # 检查价格是否变化（停牌时价格不变）
        if self.data.close[0] == self.data.close[-1]:
            if self.data.volume[0] == 0:
                # 可能是停牌
                return

        # 正常交易逻辑
        # ...
```

---

## 新股上市规则

新股上市首日有特殊规则：

| 板块 | 首日涨跌幅限制 |
|------|---------------|
| 主板 | 44% (涨), -36% (跌) |
| 科创板/创业板 | 无涨跌幅限制（前5日） |

### 过滤新股

```python
def filter_new_stocks(stock_list, trade_date, min_days=60, pro=None):
    """
    过滤掉上市不足指定天数的新股

    参数:
        stock_list: 股票代码列表
        trade_date: 当前交易日期
        min_days: 最少上市天数
        pro: tushare pro 接口

    返回:
        过滤后的股票列表
    """
    from datetime import datetime, timedelta

    trade_dt = datetime.strptime(trade_date, '%Y%m%d')
    cutoff_date = (trade_dt - timedelta(days=min_days)).strftime('%Y%m%d')

    # 获取股票上市日期
    df = pro.stock_basic(fields='ts_code,list_date')
    df = df[df['ts_code'].isin(stock_list)]

    # 过滤上市时间足够长的股票
    valid_stocks = df[df['list_date'] <= cutoff_date]['ts_code'].tolist()

    return valid_stocks
```

---

## 总结

在 A 股回测中需要注意的特殊规则：

1. **T+1**: 当日买入次日才能卖出
2. **涨跌停**: 不同板块限制不同，涨停无法买入，跌停无法卖出
3. **ST 股票**: 涨跌停 5%，建议过滤
4. **手续费**: 佣金双向 + 印花税单向（卖出）
5. **交易单位**: 买入必须是 100 股整数倍
6. **停牌**: 停牌期间无法交易
7. **新股**: 上市初期规则特殊，建议过滤

---

## 如何查询不确定的内容

遇到不确定的接口、参数或规则时，使用 Context7 查询官方文档：

### Tushare 数据接口

```
Library ID: /websites/tushare_pro_document
查询示例:
- "stk_limit 涨跌停价格接口"
- "suspend_d 停牌信息接口"
- "stock_basic 股票列表字段"
- "trade_cal 交易日历"
```

### Backtrader 回测框架

```
Library ID: /websites/backtrader_docu
查询示例:
- "Strategy 策略基类方法"
- "Order 订单状态"
- "Position 持仓管理"
- "Broker 经纪商接口"
```

### 查询方法

1. **明确需求**：确定需要查询什么（接口名、参数、返回字段）
2. **使用 Context7**：提供 Library ID 和查询关键词
3. **验证结果**：根据文档编写代码并测试
4. **记录经验**：将常用接口记录到笔记中

**原则**：宁可多查一次文档，也不要凭猜测编写代码。

---

## 参考文档

- [best-practices.md](best-practices.md) - 最佳实践和常见错误
- [data-bridge.md](data-bridge.md) - Tushare 数据对接指南
- [debugging-guide.md](debugging-guide.md) - 回测调试方法
- [known-issues.md](known-issues.md) - 已知问题和陷阱
