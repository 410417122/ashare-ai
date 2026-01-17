# Context7 文档查询指南

本文档说明如何使用 context7 查询 Tushare 和 Backtrader 的官方文档。

---

## 核心原则

**遇到任何不确定的接口、参数、字段，必须用 context7 查询官方文档。**

- 不要凭记忆猜测 API 用法
- 不要假设参数名称和返回字段
- 宁可多查一次，也不要写错误代码

---

## Tushare Pro 文档查询

### Library ID

```
/websites/tushare_pro_document
```

### 常用查询示例

| 需求 | 查询关键词 |
|------|-----------|
| 日线行情数据 | `daily 日线行情 返回字段 open high low close` |
| 股票基础信息 | `stock_basic 股票列表 ts_code name industry` |
| 财务指标 | `fina_indicator 财务指标 roe eps 参数` |
| 每日指标 | `stk_factor_pro pe pb 市盈率 市净率` |
| 复权数据 | `pro_bar adj 复权 前复权 后复权` |
| 资金流向 | `moneyflow 资金流向 主力 散户` |
| 涨跌停价格 | `stk_limit 涨跌停 up_limit down_limit` |
| 停牌信息 | `suspend_d 停牌 suspend_type` |
| 行业分类 | `index_classify 申万 行业分类` |
| 指数成分 | `index_weight 指数成分 沪深300` |

### 查询技巧

1. **使用中文关键词**：Tushare 文档是中文的，中文关键词效果更好
2. **包含字段名**：如果知道部分字段名，加入查询可以更精准
3. **说明用途**：描述你想要实现的功能

### 示例：查询日线数据接口

```
查询: "daily 日线行情 返回字段 open high low close vol amount"

期望获取:
- 接口名称和调用方式
- 输入参数说明
- 返回字段列表和含义
- 调用示例
```

---

## Backtrader 文档查询

### Library ID

```
/websites/backtrader_docu
```

### 常用查询示例

| 需求 | 查询关键词 |
|------|-----------|
| 引擎配置 | `cerebro adddata addstrategy broker` |
| 数据源 | `PandasData dataname datetime open high low close volume` |
| 策略编写 | `strategy buy sell order next __init__` |
| 技术指标 | `indicator SMA EMA RSI MACD period` |
| 布林带 | `BollingerBands indicator period devfactor` |
| 订单管理 | `order buy sell cancel valid` |
| 仓位管理 | `sizer position stake` |
| 分析器 | `analyzer SharpeRatio Returns DrawDown` |
| 手续费 | `commission CommInfoBase setcommission` |
| 绘图 | `cerebro plot style` |

### 查询技巧

1. **使用英文关键词**：Backtrader 文档是英文的
2. **包含类名/方法名**：如 `bt.indicators.RSI`
3. **描述具体功能**：如 "how to add multiple data feeds"

### 示例：查询 RSI 指标用法

```
查询: "RSI indicator period overbought oversold usage example"

期望获取:
- RSI 指标的参数说明
- 默认值
- 使用示例
- 超买超卖阈值
```

---

## 查询流程

### 步骤 1: 确定 Library ID

```python
# Tushare 接口
library_id = "/websites/tushare_pro_document"

# Backtrader 功能
library_id = "/websites/backtrader_docu"
```

### 步骤 2: 构造查询

```python
# 好的查询
query = "daily 日线行情 返回字段 open high low close vol amount"

# 不好的查询
query = "数据"  # 太模糊
```

### 步骤 3: 分析结果

- 确认接口名称正确
- 记录必需参数和可选参数
- 记录返回字段及其含义
- 注意使用限制和注意事项

---

## 何时必须查询

### 必须查询的情况

1. **使用新接口前**：第一次使用某个 Tushare 接口
2. **不确定参数时**：不确定参数名称、格式、默认值
3. **不确定返回字段时**：不知道返回的 DataFrame 有哪些列
4. **遇到报错时**：接口调用失败，需要确认正确用法
5. **实现新功能时**：需要找到合适的接口或方法

### 可以不查询的情况

1. **刚查询过**：同一会话中刚查询过的接口
2. **标准用法**：SKILL.md 中已提供的标准代码模板
3. **简单操作**：如 `df.head()`, `len(df)` 等 Pandas 基础操作

---

## 查询结果的使用

### 记录关键信息

查询后，应记录以下信息：

```python
# 接口: pro.daily()
# 参数:
#   - ts_code: 股票代码 (必需)
#   - start_date: 开始日期 (可选)
#   - end_date: 结束日期 (可选)
# 返回字段:
#   - ts_code, trade_date, open, high, low, close
#   - pre_close, change, pct_chg, vol, amount
```

### 验证查询结果

```python
# 先小范围测试
df = pro.daily(ts_code='600000.SH', start_date='20231201', end_date='20231231')
print(df.columns.tolist())  # 确认返回字段
print(df.head())  # 查看数据格式
```

---

## 常见问题

### Q: 查询没有返回想要的信息？

**A:** 尝试以下方法：
1. 换用不同的关键词
2. 使用更具体的描述
3. 包含接口名称或字段名

### Q: 返回的信息太多/太少？

**A:** 调整查询的具体程度：
- 太多：添加更具体的限定词
- 太少：使用更宽泛的描述

### Q: 文档和实际行为不一致？

**A:** 可能的原因：
1. 接口已更新，文档未同步
2. 中转接口有限制
3. 积分不足导致部分功能不可用

**解决方法：** 实际测试接口行为，以实际结果为准

---

## 快速参考卡

```
┌─────────────────────────────────────────────────────────┐
│                    Context7 快速参考                      │
├─────────────────────────────────────────────────────────┤
│  Tushare Pro 文档                                        │
│  Library ID: /websites/tushare_pro_document             │
│  语言: 中文                                              │
│  示例: "daily 日线行情 返回字段"                          │
├─────────────────────────────────────────────────────────┤
│  Backtrader 文档                                         │
│  Library ID: /websites/backtrader_docu                  │
│  语言: 英文                                              │
│  示例: "RSI indicator period usage"                     │
├─────────────────────────────────────────────────────────┤
│  原则: 不确定就查，宁可多查也不要猜                        │
└─────────────────────────────────────────────────────────┘
```

---

## 总结

1. **Tushare 查询**: `/websites/tushare_pro_document` + 中文关键词
2. **Backtrader 查询**: `/websites/backtrader_docu` + 英文关键词
3. **核心原则**: 不确定就查，不要猜测
4. **验证结果**: 查询后实际测试确认
