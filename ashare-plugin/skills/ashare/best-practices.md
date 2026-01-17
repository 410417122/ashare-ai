# A股量化策略最佳实践

本文档提供 A 股量化策略开发的最佳实践、常见反模式和系统性预防措施。遵循这些指导可以避免常见错误、前视偏差和数据污染。

---

## 目录

1. [策略开发的代码模式](#策略开发的代码模式)
2. [反模式（不要这样做）](#反模式不要这样做)
3. [前视偏差的系统性预防](#前视偏差的系统性预防)
4. [数据污染的识别和避免](#数据污染的识别和避免)
5. [常见错误对比示例](#常见错误对比示例)
6. [已知问题和陷阱](#已知问题和陷阱)
7. [调试方法论](#调试方法论)

---

## 策略开发的代码模式

### ✅ 模式1：使用向量化操作

**正确做法**：使用 pandas 的向量化操作处理整个 DataFrame。

```python
# ✅ 正确 - 向量化操作
close = df['close']
sma20 = close.rolling(20).mean()
position = close > sma20
```

**错误做法**：使用循环遍历每一行。

```python
# ❌ 错误 - 循环遍历
for i in range(len(df)):
    if df.loc[i, 'close'] > df.loc[i, 'sma20']:
        position[i] = True
```

**原因**：向量化操作速度快数百倍，代码更简洁。

---

### ✅ 模式2：使用 shift() 访问历史数据

**正确做法**：使用 `.shift()` 获取前一天的数据。

```python
# ✅ 正确 - 使用 shift()
prev_close = close.shift(1)
returns = (close - prev_close) / prev_close

# 检测金叉
sma20 = close.rolling(20).mean()
sma60 = close.rolling(60).mean()
golden_cross = (sma20 > sma60) & (sma20.shift(1) <= sma60.shift(1))
```

**错误做法**：使用索引访问。

```python
# ❌ 错误 - 使用索引
prev_close = close.iloc[-2]  # 可能导致前视偏差
```

**原因**：`.shift()` 明确表示时间偏移，避免索引错误。

---

### ✅ 模式3：数据转换的标准流程

**正确做法**：按照固定顺序转换 Tushare 数据。

```python
# ✅ 正确 - 标准转换流程
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')

# 1. 检查数据是否为空
if df.empty:
    raise ValueError("数据为空")

# 2. 日期转换
df['trade_date'] = pd.to_datetime(df['trade_date'])

# 3. 设置索引
df = df.set_index('trade_date')

# 4. 升序排列（关键！）
df = df.sort_index(ascending=True)

# 5. 列名映射
df = df.rename(columns={'vol': 'volume'})

# 6. 选择需要的列
df = df[['open', 'high', 'low', 'close', 'volume']]
```

**错误做法**：跳过某些步骤或顺序错误。

```python
# ❌ 错误 - 未升序排列
df = df.set_index('trade_date')
# 缺少 sort_index()，数据仍是降序
```

**原因**：Backtrader 要求数据升序排列，否则策略逻辑会错误。

---

### ✅ 模式4：使用自定义 Sizer 控制仓位

**正确做法**：实现自定义 Sizer 根据资金计算买入数量。

```python
# ✅ 正确 - 自定义 Sizer
class AShareSizer(bt.Sizer):
    params = (('percent', 0.95),)  # 使用95%资金

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            price = data.close[0]
            available = cash * self.p.percent
            lots = int(available / price) // 100  # 整手
            return lots * 100
        else:
            return self.broker.getposition(data).size

cerebro.addsizer(AShareSizer)
```

**错误做法**：不设置 Sizer，使用默认值。

```python
# ❌ 错误 - 默认买入1股
cerebro.addstrategy(MyStrategy)
# 10万资金只买1股，收益几乎为0
```

**原因**：Backtrader 默认买入1股，必须自定义 Sizer。

---

### ✅ 模式5：正确设置手续费

**正确做法**：使用自定义 CommInfoBase 实现 A 股手续费规则。

```python
# ✅ 正确 - A股手续费
class AShareCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.0003),  # 万三佣金
        ('stamp_duty', 0.001),   # 千一印花税
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),  # 关键：必须设置
    )

    def _getcommission(self, size, price, pseudoexec):
        turnover = abs(size) * price
        commission = max(turnover * self.p.commission, 5)  # 最低5元
        if size < 0:  # 卖出收印花税
            commission += turnover * self.p.stamp_duty
        return commission

cerebro.broker.addcommissioninfo(AShareCommission())
```

**错误做法**：使用默认手续费或 percabs 参数错误。

```python
# ❌ 错误 - percabs=False（默认）
params = (
    ('commission', 0.0003),
    ('percabs', False),  # 错误：0.0003会被理解为0.0003%
)
```

**原因**：`percabs=False` 时，0.0003 表示 0.0003% 而非 0.03%，手续费会严重偏低。

---

## 反模式（不要这样做）

### ❌ 反模式1：使用 == 比较浮点数

```python
# ❌ 错误
condition = (close == 100.0)

# ✅ 正确
import numpy as np
condition = np.isclose(close, 100.0)
# 或使用范围
condition = (close > 99.9) & (close < 100.1)
```

**原因**：浮点数精度问题导致相等比较不可靠。

---

### ❌ 反模式2：不检查空数据

```python
# ❌ 错误
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
df = df.set_index('trade_date')  # 如果df为空会报错

# ✅ 正确
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
if df.empty:
    raise ValueError("未获取到数据")
df = df.set_index('trade_date')
```

**原因**：接口可能返回空数据，必须检查。

---

### ❌ 反模式3：股票代码或日期格式错误

```python
# ❌ 错误
df = pro.daily(ts_code='600000')  # 缺少后缀
df = pro.daily(ts_code='600000.SH', start_date='2023-01-01')  # 日期格式错误

# ✅ 正确
df = pro.daily(ts_code='600000.SH', start_date='20230101')
```

**原因**：Tushare 要求特定格式。

---

### ❌ 反模式4：忘记设置 Sizer

```python
# ❌ 错误 - 收益接近0
cerebro = bt.Cerebro()
cerebro.addstrategy(MyStrategy)
cerebro.run()  # 默认买入1股

# ✅ 正确
cerebro.addsizer(AShareSizer)
```

---

### ❌ 反模式5：数据未升序排列

```python
# ❌ 错误 - Tushare数据默认降序
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
df = df.set_index('trade_date')
# 缺少 sort_index()

# ✅ 正确
df = df.sort_index(ascending=True)
```

**原因**：Backtrader 要求升序，否则策略逻辑错误。

---

## 前视偏差的系统性预防

**前视偏差（Lookahead Bias）**：使用了在决策时点不可能获得的未来信息，导致回测结果虚假乐观。

### 识别前视偏差的检查清单

#### 1. 数据对齐检查

```python
# ❌ 危险 - 财务数据未对齐
revenue = pro.income(ts_code='600000.SH', start_date='20230101', end_date='20231231')
# 财务数据的 end_date 是报告期，不是公告日期
# 使用报告期数据会导致前视偏差

# ✅ 正确 - 使用公告日期
# 查询 context7 获取带公告日期的接口
# Library ID: /websites/tushare_pro_document
# 查询: "财务数据 公告日期"
```

**原则**：财务数据必须使用公告日期（ann_date），不能使用报告期（end_date）。

---

#### 2. 指标计算检查

```python
# ❌ 危险 - 使用未来数据
sma20 = close.rolling(20).mean()
position = close > sma20  # 当天的sma20包含当天收盘价

# ✅ 正确 - 使用前一天的指标
sma20 = close.rolling(20).mean().shift(1)
position = close > sma20
```

**原则**：决策时只能使用前一天或更早的指标值。

---

#### 3. 复权数据检查

```python
# ❌ 危险 - 前复权数据会随时间变化
df = ts.pro_bar(ts_code='600000.SH', adj='qfq', start_date='20230101', end_date='20231231', api=pro)

# ✅ 正确 - 后复权数据不变
df = ts.pro_bar(ts_code='600000.SH', adj='hfq', start_date='20230101', end_date='20231231', api=pro)
```

**原则**：回测使用后复权（hfq），前复权会随时间变化导致前视偏差。

---

#### 4. 排序和筛选检查

```python
# ❌ 危险 - 使用当天数据排序
top_stocks = df.nlargest(10, 'close')  # 使用当天收盘价排序

# ✅ 正确 - 使用前一天数据排序
df['prev_close'] = df['close'].shift(1)
top_stocks = df.nlargest(10, 'prev_close')
```

**原则**：选股时使用前一天的数据。

---

### 前视偏差的常见场景

| 场景 | 错误做法 | 正确做法 |
|------|---------|---------|
| 财务数据 | 使用报告期 | 使用公告日期 |
| 技术指标 | 使用当天指标 | 使用前一天指标 |
| 复权数据 | 使用前复权 | 使用后复权 |
| 选股排序 | 使用当天数据 | 使用前一天数据 |
| 涨跌停 | 忽略涨跌停 | 检查涨跌停限制 |

---

## 数据污染的识别和避免

**数据污染**：数据处理过程中引入错误或不一致，导致回测结果失真。

### 常见数据污染场景

#### 1. 幸存者偏差

**问题**：只使用当前仍在交易的股票，忽略已退市股票。

```python
# ❌ 错误 - 只获取当前上市股票
stocks = pro.stock_basic(list_status='L')  # 只有上市股票

# ✅ 正确 - 包含退市股票
stocks = pro.stock_basic(list_status='L,D,P')  # 上市、退市、暂停上市
```

**影响**：回测结果过于乐观，因为排除了表现差的退市股票。

---

#### 2. 停牌数据缺失

**问题**：停牌期间数据缺失，但策略仍持有该股票。

```python
# ✅ 正确 - 检查停牌
def check_suspend(df):
    """检查是否停牌（成交量为0）"""
    return df['volume'] == 0

# 在策略中跳过停牌股票
if check_suspend(data):
    return  # 不交易
```

---

#### 3. 数据对齐错误

**问题**：不同频率数据对齐时引入未来信息。

```python
# ❌ 危险 - 向后填充
monthly_data = monthly_data.reindex(daily_index, method='bfill')  # 使用未来数据

# ✅ 正确 - 向前填充
monthly_data = monthly_data.reindex(daily_index, method='ffill')  # 使用历史数据
```

**原则**：只能使用 `method='ffill'`（向前填充），不能使用 `bfill`（向后填充）。

---

## 常见错误对比示例

### 示例1：数据排序

```python
# ❌ 错误 - 降序数据
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
data = bt.feeds.PandasData(dataname=df)
# 策略逻辑会完全错误

# ✅ 正确 - 升序数据
df = df.sort_index(ascending=True)
data = bt.feeds.PandasData(dataname=df)
```

---

### 示例2：列名映射

```python
# ❌ 错误 - 列名不匹配
df = df[['open', 'high', 'low', 'close', 'vol']]  # vol而非volume
data = bt.feeds.PandasData(dataname=df)
# Backtrader无法识别成交量

# ✅ 正确 - 列名映射
df = df.rename(columns={'vol': 'volume'})
df = df[['open', 'high', 'low', 'close', 'volume']]
```

---

### 示例3：日期格式

```python
# ❌ 错误 - 字符串索引
df = df.set_index('trade_date')  # trade_date仍是字符串
data = bt.feeds.PandasData(dataname=df)
# 时间序列操作会失败

# ✅ 正确 - datetime索引
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date')
```

---

### 示例4：手续费参数

```python
# ❌ 错误 - percabs=False
class AShareCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.0003),
        ('percabs', False),  # 错误！
    )
# 实际佣金是0.0003%而非0.03%

# ✅ 正确 - percabs=True
params = (
    ('commission', 0.0003),
    ('percabs', True),  # 正确
)
```

---

### 示例5：Sizer设置

```python
# ❌ 错误 - 未设置Sizer
cerebro = bt.Cerebro()
cerebro.addstrategy(MyStrategy)
cerebro.run()
# 默认买入1股，收益接近0

# ✅ 正确 - 设置Sizer
cerebro.addsizer(AShareSizer)
cerebro.run()
```

---

### 示例6：空数据检查

```python
# ❌ 错误 - 不检查空数据
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
df = df.set_index('trade_date')  # 可能报错

# ✅ 正确 - 检查空数据
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
if df.empty:
    print("警告：数据为空")
    return
df = df.set_index('trade_date')
```

---

### 示例7：股票代码格式

```python
# ❌ 错误 - 缺少后缀
df = pro.daily(ts_code='600000')

# ✅ 正确 - 带后缀
df = pro.daily(ts_code='600000.SH')
```

---

### 示例8：日期格式

```python
# ❌ 错误 - 带分隔符
df = pro.daily(ts_code='600000.SH', start_date='2023-01-01')

# ✅ 正确 - 无分隔符
df = pro.daily(ts_code='600000.SH', start_date='20230101')
```

---

### 示例9：T+1规则

```python
# ❌ 错误 - 当日买入当日卖出
class MyStrategy(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy()
        else:
            self.sell()  # 违反T+1

# ✅ 正确 - 实现T+1
class T1Strategy(bt.Strategy):
    def __init__(self):
        self.buy_bar = None

    def next(self):
        if self.position:
            if len(self) - self.buy_bar >= 1:  # 持有至少1天
                if self.should_sell():
                    self.sell()
        else:
            if self.should_buy():
                self.buy()
                self.buy_bar = len(self)
```

---

### 示例10：复权数据

```python
# ❌ 错误 - 不复权数据
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
# 除权除息导致价格跳空

# ✅ 正确 - 后复权数据
df = ts.pro_bar(ts_code='600000.SH', adj='hfq', start_date='20230101', end_date='20231231', api=pro)
```

---

## 已知问题和陷阱

以下是使用 Tushare + Backtrader 进行 A 股回测时的已知问题（详细内容参考 [known-issues.md](known-issues.md)）：

### 数据相关（6个问题）

1. **北向资金数据不可用**：`moneyflow_hsgt` 接口从2024年起不再提供数据
2. **数据排序问题**：Tushare 默认降序，Backtrader 要求升序
3. **成交量列名不一致**：Tushare 用 `vol`，Backtrader 用 `volume`
4. **日期格式问题**：Tushare 返回字符串，需转换为 datetime
5. **复权数据获取**：`pro.daily()` 不复权，需用 `ts.pro_bar(adj='hfq')`
6. **Tushare 积分限制**：部分高级接口需要较高积分

### 回测相关（6个问题）

7. **T+1规则未处理**：Backtrader 默认不处理 T+1
8. **涨跌停未处理**：默认不考虑涨跌停限制
9. **手续费计算不准确**：默认设置不符合 A 股规则
10. **整手交易未处理**：默认可买入任意数量
11. **停牌数据缺失**：停牌期间数据缺失
12. **新股数据异常**：新股上市初期价格波动大

### 代码相关（7个问题）

13. **股票代码格式错误**：必须带后缀（.SH 或 .SZ）
14. **日期格式错误**：必须是 YYYYMMDD 格式
15. **空数据未检查**：接口可能返回空 DataFrame
16. **默认买入1股**：Backtrader 默认 Sizer 买入1股
17. **percabs参数错误**：影响手续费计算
18. **ST股票处理**：ST股票涨跌停5%，建议过滤
19. **调用频率限制**：批量获取数据可能被限流

---

## 调试方法论

### 核心原则

**代码能运行 ≠ 逻辑正确**

回测调试分两层：
1. **技术层**：代码能跑通，没有报错
2. **逻辑层**：回测逻辑符合预期，结果合理

很多 bug 不会报错，但会导致回测结果完全失真。

---

### 调试流程

#### 第一步：验证数据

```python
# 1. 检查数据是否为空
print(f'数据行数: {len(df)}')
if df.empty:
    raise ValueError("数据为空")

# 2. 检查数据排序（必须升序）
print(f'前5个日期: {df.index[:5].tolist()}')

# 3. 检查列名
print(f'列名: {df.columns.tolist()}')

# 4. 检查数据类型
print(df.dtypes)

# 5. 检查空值
print(f'空值数量: {df.isnull().sum().sum()}')
```

---

#### 第二步：验证交易行为

```python
class DebugStrategy(bt.Strategy):
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'买入: {self.data.datetime.date(0)} '
                      f'价格={order.executed.price:.2f} '
                      f'数量={order.executed.size}')
            else:
                print(f'卖出: {self.data.datetime.date(0)} '
                      f'价格={order.executed.price:.2f}')
```

**关键检查点**：
- 买入数量是否合理？（不应该是1股）
- 买入金额是否接近可用资金？
- 交易日期间隔是否符合 T+1？

---

#### 第三步：验证手续费

```python
# 单独测试手续费类
comm = AShareCommission()

test_cases = [
    (10000, 10.0, '买入10万'),
    (-10000, 10.0, '卖出10万'),
    (100, 10.0, '买入1000元'),
]

for size, price, desc in test_cases:
    fee = comm._getcommission(size, price, False)
    turnover = abs(size) * price
    expected = max(turnover * 0.0003, 5)
    if size < 0:
        expected += turnover * 0.001
    status = 'OK' if abs(fee - expected) < 0.01 else 'FAIL'
    print(f'{desc}: 计算={fee:.2f}, 预期={expected:.2f} [{status}]')
```

---

#### 第四步：验证回测结果

```python
# 检查回测结果是否合理
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

### 调试检查清单

在提交回测代码前，逐项检查：

#### 数据层
- [ ] 数据不为空
- [ ] 数据升序排列
- [ ] 列名正确（vol → volume）
- [ ] 日期格式正确（datetime 索引）
- [ ] 无空值

#### 配置层
- [ ] 设置了 Sizer（不是默认的1股）
- [ ] 设置了手续费类
- [ ] `percabs=True`（如果使用小数形式的佣金率）
- [ ] 初始资金设置正确

#### 逻辑层
- [ ] 买入数量合理（接近可用资金）
- [ ] 手续费计算正确
- [ ] T+1 规则正确实现（如需要）
- [ ] 涨跌停处理正确（如需要）

#### 结果层
- [ ] 收益率在合理范围
- [ ] 夏普比率有值
- [ ] 最大回撤有值
- [ ] 交易次数合理

---

## 如何查询不确定的内容

遇到不确定的接口、参数或方法时，使用 Context7 查询官方文档：

### Tushare 数据接口

```
Library ID: /websites/tushare_pro_document
查询示例:
- "daily 接口参数"
- "income 财务数据公告日期"
- "stk_limit 涨跌停价格"
```

### Backtrader 回测框架

```
Library ID: /websites/backtrader_docu
查询示例:
- "CommInfoBase 参数说明"
- "Sizer 自定义"
- "Analyzer 使用方法"
```

### pandas 数据处理

```
查询示例:
- "pandas rolling 移动窗口"
- "pandas shift 时间偏移"
- "pandas reindex 重新索引"
```

**原则**：宁可多查一次文档，也不要写错误代码。

---

## 参考文档

- [SKILL.md](SKILL.md) - 快速开始和核心概念
- [data-reference.md](data-reference.md) - Tushare 数据参考
- [backtesting-reference.md](backtesting-reference.md) - Backtrader 回测参考
- [dataframe-reference.md](dataframe-reference.md) - 数据处理工具
- [factor-examples.md](factor-examples.md) - 策略示例库
- [ashare-rules.md](ashare-rules.md) - A股特殊交易规则
- [data-bridge.md](data-bridge.md) - 数据对接指南
- [debugging-guide.md](debugging-guide.md) - 调试指南
- [known-issues.md](known-issues.md) - 已知问题详细列表
