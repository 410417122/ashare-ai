# 回测调试指南

本文档总结了调试 Backtrader 回测时的方法论和经验，帮助快速定位和修复问题。

---

## 核心原则

**代码能运行 ≠ 逻辑正确**

回测调试分两层：
1. **技术层**：代码能跑通，没有报错
2. **逻辑层**：回测逻辑符合预期，结果合理

很多 bug 不会报错，但会导致回测结果完全失真。

---

## 调试方法论

### 第一步：验证数据

在运行回测前，先验证数据是否正确。

```python
# 1. 检查数据是否为空
print(f'数据行数: {len(df)}')
if df.empty:
    raise ValueError("数据为空")

# 2. 检查数据排序（必须升序）
print(f'前5个日期: {df.index[:5].tolist()}')
# 应该是从早到晚，如 2023-01-03, 2023-01-04, ...

# 3. 检查列名
print(f'列名: {df.columns.tolist()}')
# 必须包含: open, high, low, close, volume

# 4. 检查数据类型
print(df.dtypes)
# 价格和成交量应该是 float64

# 5. 检查空值
print(f'空值数量: {df.isnull().sum().sum()}')
```

### 第二步：验证交易行为

运行回测时，打印每笔交易的详细信息。

```python
class DebugStrategy(bt.Strategy):
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'买入: {self.data.datetime.date(0)} '
                      f'价格={order.executed.price:.2f} '
                      f'数量={order.executed.size} '
                      f'金额={order.executed.price * order.executed.size:.2f}')
            else:
                print(f'卖出: {self.data.datetime.date(0)} '
                      f'价格={order.executed.price:.2f} '
                      f'数量={order.executed.size}')
```

**关键检查点：**
- 买入数量是否合理？（不应该是 1 股）
- 买入金额是否接近可用资金？
- 交易日期间隔是否符合 T+1？

### 第三步：验证手续费

单独测试手续费类，确保计算正确。

```python
# 创建手续费实例
comm = AShareCommission()

# 测试用例
test_cases = [
    (10000, 10.0, '买入10万'),   # 大额买入
    (-10000, 10.0, '卖出10万'),  # 大额卖出
    (100, 10.0, '买入1000元'),   # 小额买入（测试最低5元）
    (-100, 10.0, '卖出1000元'),  # 小额卖出
]

for size, price, desc in test_cases:
    fee = comm._getcommission(size, price, False)

    # 手动计算预期值
    turnover = abs(size) * price
    expected = max(turnover * 0.0003, 5)  # 佣金
    if size < 0:
        expected += turnover * 0.001  # 印花税

    status = 'OK' if abs(fee - expected) < 0.01 else 'FAIL'
    print(f'{desc}: 计算={fee:.2f}, 预期={expected:.2f} [{status}]')
```

### 第四步：验证回测结果

检查回测结果是否合理。

```python
# 运行回测
results = cerebro.run()
strat = results[0]

# 基本指标
initial = 100000
final = cerebro.broker.getvalue()
total_return = (final / initial - 1) * 100

print(f'初始资金: {initial:,.2f}')
print(f'最终资金: {final:,.2f}')
print(f'总收益率: {total_return:.2f}%')

# 合理性检查
if abs(total_return) < 0.01:
    print('警告: 收益率接近0，可能存在问题')
    print('  - 检查 Sizer 是否正确设置')
    print('  - 检查是否有交易发生')

if total_return > 1000:
    print('警告: 收益率异常高，可能存在问题')
    print('  - 检查是否有前视偏差')
    print('  - 检查数据是否正确')
```

---

## 常见问题排查

### 问题1：收益率接近 0

**症状：** 10万资金，回测2年，收益只有几元钱

**排查步骤：**
1. 打印交易记录，检查买入数量
2. 如果买入数量是 1 股，说明没有设置 Sizer

**解决方案：**
```python
cerebro.addsizer(AShareSizer)
```

### 问题2：手续费计算错误

**症状：** 手续费远低于预期

**排查步骤：**
1. 打印 `comm.p.commission` 检查参数值
2. 如果值是 `2.9e-06` 而不是 `0.0003`，说明 `percabs` 参数问题

**解决方案：**
```python
params = (
    ('commission', 0.0003),
    ('percabs', True),  # 关键
)
```

### 问题3：夏普比率为 None 或异常

**症状：** `sharperatio: None` 或 `-500`

**排查步骤：**
1. 检查数据时间跨度是否足够（至少1年）
2. 检查是否有足够的交易

**解决方案：**
- 使用 `SharpeRatio_A` 替代 `SharpeRatio`
- 确保数据跨度足够

### 问题4：最大回撤为 0

**症状：** `max.drawdown: 0.00%`

**排查步骤：**
1. 检查是否有交易发生
2. 检查资金曲线是否有波动

**可能原因：**
- 没有交易发生
- 买入数量太小，波动可忽略

---

## 调试检查清单

在提交回测代码前，逐项检查：

### 数据层
- [ ] 数据不为空
- [ ] 数据升序排列
- [ ] 列名正确（vol → volume）
- [ ] 日期格式正确（datetime 索引）
- [ ] 无空值

### 配置层
- [ ] 设置了 Sizer（不是默认的 1 股）
- [ ] 设置了手续费类
- [ ] `percabs=True`（如果使用小数形式的佣金率）
- [ ] 初始资金设置正确

### 逻辑层
- [ ] 买入数量合理（接近可用资金）
- [ ] 手续费计算正确
- [ ] T+1 规则正确实现（如需要）
- [ ] 涨跌停处理正确（如需要）

### 结果层
- [ ] 收益率在合理范围
- [ ] 夏普比率有值
- [ ] 最大回撤有值
- [ ] 交易次数合理

---

## 调试代码模板

```python
import tushare as ts
import backtrader as bt
import pandas as pd

# ========== 1. 数据获取和验证 ==========
pro = ts.pro_api('占位符')
pro._DataApi__token = 'YOUR_TOKEN'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20241231')

# 数据验证
print('=== 数据验证 ===')
print(f'行数: {len(df)}')
print(f'列名: {df.columns.tolist()}')
print(f'日期范围: {df["trade_date"].min()} ~ {df["trade_date"].max()}')

# 数据转换
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()
df = df.rename(columns={'vol': 'volume'})
df = df[['open', 'high', 'low', 'close', 'volume']]

print(f'转换后排序: {df.index[:3].tolist()}')

# ========== 2. 手续费验证 ==========
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
        commission = turnover * self.p.commission
        if commission < 5:
            commission = 5
        if size < 0:
            commission += turnover * self.p.stamp_duty
        return commission

print('\n=== 手续费验证 ===')
comm = AShareCommission()
print(f'买入10万: {comm._getcommission(10000, 10.0, False):.2f} (预期: 30.00)')
print(f'卖出10万: {comm._getcommission(-10000, 10.0, False):.2f} (预期: 130.00)')

# ========== 3. Sizer 验证 ==========
class AShareSizer(bt.Sizer):
    params = (('percent', 0.95),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            price = data.close[0]
            available = cash * self.p.percent
            lots = int(available / price) // 100
            return lots * 100
        else:
            return self.broker.getposition(data).size

# ========== 4. 策略（带调试输出） ==========
class DebugStrategy(bt.Strategy):
    params = (('period', 20),)

    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.period)
        self.order = None
        self.trade_count = 0

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.data.close[0] > self.sma[0]:
                self.order = self.buy()
        else:
            if self.data.close[0] < self.sma[0]:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.trade_count += 1
            if order.isbuy():
                print(f'[{self.trade_count}] 买入: {self.data.datetime.date(0)} '
                      f'价格={order.executed.price:.2f} 数量={order.executed.size}')
            else:
                print(f'[{self.trade_count}] 卖出: {self.data.datetime.date(0)} '
                      f'价格={order.executed.price:.2f}')
        self.order = None

# ========== 5. 运行回测 ==========
print('\n=== 运行回测 ===')
cerebro = bt.Cerebro()
cerebro.addstrategy(DebugStrategy)
cerebro.addsizer(AShareSizer)
cerebro.broker.addcommissioninfo(AShareCommission())
cerebro.adddata(bt.feeds.PandasData(dataname=df))
cerebro.broker.setcash(100000)

cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

initial = cerebro.broker.getvalue()
results = cerebro.run()
final = cerebro.broker.getvalue()
strat = results[0]

# ========== 6. 结果验证 ==========
print('\n=== 结果验证 ===')
print(f'初始资金: {initial:,.2f}')
print(f'最终资金: {final:,.2f}')
print(f'总收益: {(final/initial - 1)*100:.2f}%')

sharpe = strat.analyzers.sharpe.get_analysis()
returns = strat.analyzers.returns.get_analysis()
dd = strat.analyzers.drawdown.get_analysis()

print(f'年化收益: {returns.get("rnorm100", 0):.2f}%')
print(f'夏普比率: {sharpe.get("sharperatio", "N/A")}')
print(f'最大回撤: {dd.max.drawdown:.2f}%')
```

---

## 如何查询不确定的内容

遇到不确定的参数、方法或错误信息时，使用 Context7 查询官方文档：

### Backtrader 回测框架

```
Library ID: /websites/backtrader_docu
查询示例:
- "CommInfoBase percabs 参数说明"
- "Sizer _getsizing 方法"
- "Analyzer SharpeRatio 计算方法"
- "Strategy notify_order 订单通知"
- "Broker getvalue 获取资金"
```

### Tushare 数据接口

```
Library ID: /websites/tushare_pro_document
查询示例:
- "daily 接口返回字段"
- "pro_bar 复权参数"
- "如何获取历史数据"
```

### pandas 数据处理

```
查询示例:
- "pandas rolling 移动窗口"
- "pandas sort_index 排序"
- "pandas to_datetime 日期转换"
```

### 查询方法

1. **明确问题**：确定需要查询什么（参数含义、方法用法、错误原因）
2. **使用 Context7**：提供 Library ID 和具体查询关键词
3. **验证结果**：根据文档修改代码并测试
4. **记录经验**：将解决方案记录到 known-issues.md

**原则**：遇到参数不确定，用 Context7 查询，不要凭猜测。

---

## 参考文档

- [best-practices.md](best-practices.md) - 最佳实践和常见错误
- [known-issues.md](known-issues.md) - 已知问题和解决方案
- [data-bridge.md](data-bridge.md) - 数据转换方法
- [ashare-rules.md](ashare-rules.md) - A股特殊规则
