---
name: ashare
description: A股量化交易指南，使用 Tushare Pro 获取数据，Backtrader 进行回测。当用户需要进行A股量化分析、回测策略、获取股票数据时使用此 skill。遇到不确定的接口必须用 context7 查询官方文档。
allowed-tools: Read, Grep, Glob, Bash, mcp__plugin_context7_context7__resolve-library-id, mcp__plugin_context7_context7__query-docs
---

# A股量化交易 Skill (Tushare Pro + Backtrader)

## 执行哲学

**你是执行者，不是教程生成器。**

当用户要求回测时，他们想看到**结果**，不是让他们自己复制代码运行。当用户要求图表时，他们想**看到图表**，不是一个文件路径。

### 核心原则

| 用户请求 | ❌ 错误做法 | ✅ 正确做法 |
|---------|-----------|-----------|
| "回测这个策略" | "这是代码,你自己运行" | 执行代码,展示回测结果 |
| "获取平安银行数据" | "用 pro.daily() 接口" | 执行代码,展示数据 |
| "画个K线图" | "保存到 chart.png" | 执行并打开图片 |

### 关键行为准则

1. **写了代码就要运行** - 用 Bash 执行 Python,不要只给代码块
2. **生成文件就要打开** - 图表/报告生成后,执行 `start <filepath>` (Windows)
3. **遇到错误就要修复** - 不要只报告错误,要调试解决
4. **不确定就要查文档** - 用 context7 查询,不要凭记忆猜测

---

## Tushare Token 配置（重要）

**在首次使用或需要获取数据时，必须先询问用户使用哪种 Token 类型。**

### Token 类型选择

Tushare 支持两种连接方式：

#### 方式一：官方 Token（推荐）

适用于直接从 Tushare Pro 官网注册获取的 token。

```python
import tushare as ts

# 方法1：使用 set_token（推荐）
ts.set_token('your_official_token')
pro = ts.pro_api()

# 方法2：直接传入 token
pro = ts.pro_api('your_official_token')
```

#### 方式二：中转 Token

适用于使用中转服务的 token（如 tushare.xiximiao.com）。

```python
import tushare as ts

pro = ts.pro_api('占位符')
pro._DataApi__token = '你的中转Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'
```

### 使用流程

1. **首次使用时询问**：
   ```
   请问您使用的是：
   1. Tushare 官方 Token（从 tushare.pro 注册获取）
   2. 中转 Token（如 tushare.xiximiao.com）

   请告诉我您使用的是哪种 Token，以及您的 Token 值。
   ```

2. **根据用户选择使用对应的初始化代码**

3. **后续使用时**：
   - 如果用户已经配置过，直接使用对应的方式
   - 如果不确定，再次询问用户

### 注意事项

- 不要假设用户使用哪种方式，必须明确询问
- 官方 Token 和中转 Token 的初始化代码不同，不可混用
- 中转 Token 需要额外设置 `http_url`

---

## 核心原则：不确定就查文档

**这是最重要的原则：遇到任何不确定的接口、参数、字段,必须用 context7 查询官方文档。**

### Tushare 接口查询

```
Library ID: /websites/tushare_pro_document

查询示例：
- "daily 日线行情 返回字段 open high low close"
- "fina_indicator 财务指标 roe eps 参数"
- "stock_basic 股票列表 industry market"
- "stk_factor_pro pe pb 市盈率 市净率"
- "moneyflow 资金流向 buy_sm buy_md"
```

### Backtrader 功能查询

```
Library ID: /websites/backtrader_docu

查询示例：
- "cerebro adddata PandasData dataframe"
- "strategy buy sell order next"
- "indicator SMA EMA RSI MACD period"
- "analyzer SharpeRatio Returns DrawDown"
- "broker setcommission commission"
```

### 查询时机

- 使用任何 tushare 接口前,先确认参数和返回字段
- 使用任何 backtrader 功能前,先确认用法
- 遇到报错时,查询正确用法
- **宁可多查一次,也不要写错误代码**

---

## 快速开始

### 完整回测示例

```python
import tushare as ts
import backtrader as bt
import pandas as pd

# ========== 1. 初始化 Tushare ==========
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的中转Token'  # 替换为实际 token
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# ========== 2. 定义策略 ==========
class SMAStrategy(bt.Strategy):
    """简单移动平均策略"""
    params = (('period', 20),)

    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.period)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.sma[0]:
                self.buy()
        else:
            if self.data.close[0] < self.sma[0]:
                self.sell()

# ========== 3. 获取数据 ==========
df = pro.daily(ts_code='600000.SH', start_date='20220101', end_date='20231231')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()
df = df.rename(columns={'vol': 'volume'})
df = df[['open', 'high', 'low', 'close', 'volume']]

# ========== 4. 运行回测 ==========
cerebro = bt.Cerebro()
cerebro.addstrategy(SMAStrategy)
cerebro.adddata(bt.feeds.PandasData(dataname=df))
cerebro.broker.setcash(100000)
cerebro.broker.setcommission(commission=0.001)

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

# 绘图
cerebro.plot()
```

---

## 核心代码片段

### Tushare 初始化
```python
import tushare as ts
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'
```

### 数据转换为 Backtrader 格式
```python
import pandas as pd
import backtrader as bt

# 获取数据
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')

# 转换
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()
df = df.rename(columns={'vol': 'volume'})
df = df[['open', 'high', 'low', 'close', 'volume']]

# 创建数据源
data = bt.feeds.PandasData(dataname=df)
```

详细的数据对接代码见 [data-bridge.md](data-bridge.md)。

### A股手续费设置
```python
import backtrader as bt

class AShareCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.0003),   # 佣金万三
        ('stamp_duty', 0.001),    # 印花税千一
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),  # 关键参数
    )

    def _getcommission(self, size, price, pseudoexec):
        turnover = abs(size) * price
        commission = max(turnover * self.p.commission, 5)  # 最低5元
        if size < 0:  # 卖出收印花税
            commission += turnover * self.p.stamp_duty
        return commission

cerebro.broker.addcommissioninfo(AShareCommission())
```

详细说明见 [ashare-rules.md](ashare-rules.md)。

---

## 参考文档索引

### 核心文档

| 文档 | 说明 | 何时使用 |
|------|------|---------|
| [data-reference.md](data-reference.md) | Tushare 数据参考和查询指南 | 需要获取数据时 |
| [factor-examples.md](factor-examples.md) | 策略示例库和开发方法论 | 开发策略时 |
| [dataframe-reference.md](dataframe-reference.md) | 数据处理工具函数 | 需要数据处理时 |
| [factor-analysis-reference.md](factor-analysis-reference.md) | 因子分析工具 | 进行因子研究时 |
| [machine-learning-reference.md](machine-learning-reference.md) | 机器学习集成指南 | 使用ML开发策略时 |
| [best-practices.md](best-practices.md) | 最佳实践和反模式 | 避免常见错误 |

### 技术文档

| 文档 | 说明 |
|------|------|
| [data-bridge.md](data-bridge.md) | Tushare → Backtrader 数据对接详解 |
| [ashare-rules.md](ashare-rules.md) | A股特殊规则 (T+1, 涨跌停等) |
| [context7-guide.md](context7-guide.md) | 如何用 context7 查询官方文档 |
| [known-issues.md](known-issues.md) | 已知问题和陷阱 |
| [debugging-guide.md](debugging-guide.md) | 回测调试指南和方法论 |

---

## 文档使用指南

### 渐进式信息披露

本 skill 采用**渐进式信息披露**架构,避免一次性加载所有信息：

```
用户请求
    ↓
SKILL.md (本文档)
├── 快速开始示例
├── 核心原则：不确定就查文档
└── 参考文档索引
    ↓
专题文档 (按需阅读)
├── data-reference.md - 数据查询方法
├── factor-examples.md - 策略开发方法论
├── dataframe-reference.md - 数据处理工具
├── factor-analysis-reference.md - 因子分析
├── machine-learning-reference.md - ML 集成
└── best-practices.md - 最佳实践
    ↓
Context7 查询 (按需查询)
├── Tushare 官方文档
├── Backtrader 官方文档
├── pandas 文档
└── scikit-learn 文档
```

### 使用建议

1. **首次使用**：阅读本文档的快速开始示例
2. **开发策略**：参考 factor-examples.md 的策略模板
3. **获取数据**：参考 data-reference.md 的查询指南
4. **遇到问题**：查看 known-issues.md 和 debugging-guide.md
5. **不确定时**：使用 context7 查询官方文档

---

## 已知问题速查

- **北向资金数据 (moneyflow_hsgt)**: 2024年起不可用,不要使用
- **Tushare 积分限制**: 部分高级接口需要较高积分
- **数据排序**: Backtrader 要求数据升序排列,Tushare 返回降序
- **手续费设置**: 必须设置 `percabs=True`,否则手续费计算错误

详见 [known-issues.md](known-issues.md)

---

## 语言

**使用用户的语言回复。** 用户用中文提问,用中文回答。
