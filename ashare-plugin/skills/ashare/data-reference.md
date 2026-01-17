# Tushare 数据参考和查询指南

## 核心原则

**不要试图记住所有数据接口和字段,而是学会如何查询官方文档。**

本文档提供：
1. 如何使用 Context7 查询 Tushare 数据
2. 常用数据集速查表
3. 数据发现流程
4. 典型数据获取示例

---

## 如何查询 Tushare 数据

### Context7 查询方法

```
Library ID: /websites/tushare_pro_document

查询模板：
"[接口名] [功能描述] [关键字段]"

示例：
- "daily 日线行情 返回字段 open high low close"
- "fina_indicator 财务指标 roe eps 参数"
- "stock_basic 股票列表 industry market"
- "stk_factor_pro pe pb 市盈率 市净率"
- "moneyflow 资金流向 buy_sm buy_md"
- "income 利润表 revenue profit"
- "balancesheet 资产负债表 total_assets"
- "cashflow 现金流量表 operating_cash_flow"
```

### 查询时机

- ✅ 使用任何接口前,先查询参数和返回字段
- ✅ 不确定字段含义时,查询字段说明
- ✅ 遇到报错时,查询正确用法
- ✅ 需要新数据时,查询是否有对应接口

---

## 常用数据集速查表

### 行情数据

| 数据类型 | 接口名 | 主要字段 | 查询关键词 |
|---------|--------|---------|-----------|
| 日线行情 | daily | open, high, low, close, vol | "daily 日线行情" |
| 周线行情 | weekly | open, high, low, close, vol | "weekly 周线行情" |
| 月线行情 | monthly | open, high, low, close, vol | "monthly 月线行情" |
| 复权因子 | adj_factor | adj_factor | "adj_factor 复权因子" |
| 停复牌信息 | suspend | suspend_date, resume_date | "suspend 停复牌" |

### 基本信息

| 数据类型 | 接口名 | 主要字段 | 查询关键词 |
|---------|--------|---------|-----------|
| 股票列表 | stock_basic | ts_code, name, industry, market | "stock_basic 股票列表" |
| 交易日历 | trade_cal | cal_date, is_open | "trade_cal 交易日历" |
| 股票曾用名 | namechange | ts_code, name, start_date | "namechange 曾用名" |
| 沪深股通成分股 | hs_const | ts_code, hs_type | "hs_const 沪深股通" |

### 财务数据

| 数据类型 | 接口名 | 主要字段 | 查询关键词 |
|---------|--------|---------|-----------|
| 利润表 | income | revenue, n_income, operate_profit | "income 利润表" |
| 资产负债表 | balancesheet | total_assets, total_liab, total_equity | "balancesheet 资产负债表" |
| 现金流量表 | cashflow | c_fr_sale_sg, n_cashflow_act | "cashflow 现金流量表" |
| 财务指标 | fina_indicator | roe, eps, pe, pb | "fina_indicator 财务指标" |
| 业绩预告 | forecast | type, p_change_min, p_change_max | "forecast 业绩预告" |
| 业绩快报 | express | revenue, operate_profit, total_profit | "express 业绩快报" |

### 市场参考数据

| 数据类型 | 接口名 | 主要字段 | 查询关键词 |
|---------|--------|---------|-----------|
| 每日指标 | daily_basic | pe, pb, ps, dv_ratio, turnover_rate | "daily_basic 每日指标" |
| 估值因子 | stk_factor_pro | pe, pb, ps, pcf, dv_ratio | "stk_factor_pro 估值因子" |
| 资金流向 | moneyflow | buy_sm, buy_md, buy_lg, buy_elg | "moneyflow 资金流向" |
| 融资融券 | margin | rzye, rqye, rzmre, rqmcl | "margin 融资融券" |
| 龙虎榜 | top_list | amount, net_amount, buy, sell | "top_list 龙虎榜" |

### 指数数据

| 数据类型 | 接口名 | 主要字段 | 查询关键词 |
|---------|--------|---------|-----------|
| 指数基本信息 | index_basic | ts_code, name, market, category | "index_basic 指数基本信息" |
| 指数日线行情 | index_daily | open, high, low, close, vol | "index_daily 指数日线" |
| 指数成分股 | index_weight | ts_code, weight | "index_weight 指数成分股" |

---

## 数据发现流程

### 步骤1：确定需求

明确你需要什么数据：
- 价格数据？财务数据？资金流向？
- 日频？周频？月频？
- 需要哪些字段？

### 步骤2：查询 Context7

使用 Context7 查询 Tushare 官方文档：

```python
# 示例：需要市盈率数据
# 查询："Tushare 如何获取市盈率数据？"
# 或："stk_factor_pro pe pb 市盈率 市净率"
```

### 步骤3：根据返回结果使用接口

```python
import tushare as ts

# 初始化
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 根据 Context7 返回的接口和参数获取数据
df = pro.stk_factor_pro(ts_code='600000.SH', start_date='20230101', end_date='20231231')
```

### 步骤4：验证数据

```python
# 检查数据
print(f"获取到 {len(df)} 条数据")
print(df.head())
print(df.columns.tolist())  # 查看所有字段
print(df.isnull().sum())    # 检查缺失值
```

---

## 典型数据获取示例

### 示例1：获取日线行情数据

```python
import tushare as ts
import pandas as pd

# 初始化
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取单只股票日线数据
df = pro.daily(
    ts_code='600000.SH',      # 股票代码
    start_date='20230101',    # 开始日期
    end_date='20231231'       # 结束日期
)

# 数据处理
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.sort_values('trade_date')

print(f"获取到 {len(df)} 条数据")
print(df.head())
```

### 示例2：获取多只股票数据

```python
import tushare as ts
import pandas as pd

# 初始化
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取股票列表
stocks = ['600000.SH', '000001.SZ', '600519.SH']

# 批量获取数据
all_data = []
for ts_code in stocks:
    df = pro.daily(ts_code=ts_code, start_date='20230101', end_date='20231231')
    df['ts_code'] = ts_code  # 添加股票代码列
    all_data.append(df)

# 合并数据
combined_df = pd.concat(all_data, ignore_index=True)
print(f"总共获取到 {len(combined_df)} 条数据")
```

### 示例3：获取财务指标数据

```python
import tushare as ts

# 初始化
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取财务指标
df = pro.fina_indicator(
    ts_code='600000.SH',
    start_date='20220101',
    end_date='20231231',
    fields='ts_code,end_date,roe,eps,bps,current_ratio,quick_ratio'
)

print(df)
```

### 示例4：获取每日估值指标

```python
import tushare as ts

# 初始化
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取每日估值指标
df = pro.daily_basic(
    ts_code='600000.SH',
    start_date='20230101',
    end_date='20231231',
    fields='ts_code,trade_date,pe,pb,ps,dv_ratio,turnover_rate'
)

print(df)
```

### 示例5：获取资金流向数据

```python
import tushare as ts

# 初始化
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取资金流向
df = pro.moneyflow(
    ts_code='600000.SH',
    start_date='20230101',
    end_date='20231231'
)

# 资金流向字段说明：
# buy_sm: 小单买入金额（万元）
# buy_md: 中单买入金额（万元）
# buy_lg: 大单买入金额（万元）
# buy_elg: 特大单买入金额（万元）
# sell_sm, sell_md, sell_lg, sell_elg: 对应卖出金额

print(df)
```

### 示例6：获取融资融券数据

```python
import tushare as ts

# 初始化
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取融资融券数据
df = pro.margin(
    ts_code='600000.SH',
    start_date='20230101',
    end_date='20231231'
)

# 融资融券字段说明：
# rzye: 融资余额（元）
# rqye: 融券余额（元）
# rzmre: 融资买入额（元）
# rqmcl: 融券卖出量（股）

print(df)
```

### 示例7：获取指数成分股

```python
import tushare as ts

# 初始化
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取沪深300成分股
df = pro.index_weight(
    index_code='000300.SH',  # 沪深300
    start_date='20230101',
    end_date='20231231'
)

print(f"沪深300成分股数量: {df['con_code'].nunique()}")
print(df.head())
```

### 示例8：获取行业分类

```python
import tushare as ts

# 初始化
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取股票列表（包含行业信息）
df = pro.stock_basic(
    exchange='',
    list_status='L',
    fields='ts_code,name,industry,market'
)

# 按行业统计
industry_count = df['industry'].value_counts()
print("各行业股票数量：")
print(industry_count)
```

---

## 数据质量说明

### 更新频率

| 数据类型 | 更新频率 | 延迟 |
|---------|---------|------|
| 日线行情 | 每日 | 当日收盘后 |
| 财务数据 | 季度 | 财报发布后 |
| 每日指标 | 每日 | 当日收盘后 |
| 资金流向 | 每日 | 当日收盘后 |
| 融资融券 | 每日 | 次日 |

### 历史数据范围

- **日线行情**: 通常可追溯到股票上市日
- **财务数据**: 通常可追溯到2000年左右
- **资金流向**: 通常可追溯到2010年左右
- **融资融券**: 通常可追溯到2010年左右

### 数据缺失处理

```python
import pandas as pd

# 检查缺失值
print(df.isnull().sum())

# 填充缺失值
df = df.fillna(method='ffill')  # 前向填充
df = df.fillna(0)               # 填充为0
df = df.dropna()                # 删除缺失行
```

---

## 常见问题

### Q1: 如何知道某个接口需要什么参数？

**A**: 使用 Context7 查询官方文档。

```
查询："[接口名] 参数 返回字段"
示例："daily 参数 返回字段"
```

### Q2: 如何知道某个字段的含义？

**A**: 使用 Context7 查询字段说明。

```
查询："[接口名] [字段名] 含义"
示例："fina_indicator roe 含义"
```

### Q3: 如何获取某个特定数据？

**A**: 先确定需求,再查询 Context7。

```
示例需求："我需要获取股票的市盈率数据"
查询："Tushare 市盈率 pe 接口"
```

### Q4: 数据获取失败怎么办？

**A**: 检查以下几点：
1. Token 是否正确
2. 接口参数是否正确
3. 是否有权限访问该接口（积分限制）
4. 网络连接是否正常

### Q5: 如何提高数据获取效率？

**A**:
1. 使用 `fields` 参数只获取需要的字段
2. 批量获取时添加延时,避免频率限制
3. 缓存已获取的数据,避免重复请求

```python
import time

# 批量获取时添加延时
for ts_code in stocks:
    df = pro.daily(ts_code=ts_code, start_date='20230101', end_date='20231231')
    all_data.append(df)
    time.sleep(0.3)  # 延时300ms
```

---

## 数据对接到 Backtrader

获取数据后,需要转换为 Backtrader 格式。详见 [data-bridge.md](data-bridge.md)。

```python
import backtrader as bt
import pandas as pd

# 获取数据
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')

# 转换为 Backtrader 格式
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()
df = df.rename(columns={'vol': 'volume'})
df = df[['open', 'high', 'low', 'close', 'volume']]

# 创建 Backtrader 数据源
data = bt.feeds.PandasData(dataname=df)
```

---

## 总结

**核心原则**：
1. ✅ 不要试图记住所有接口和字段
2. ✅ 学会使用 Context7 查询官方文档
3. ✅ 遵循"确定需求 → 查询文档 → 使用接口 → 验证数据"流程
4. ✅ 遇到问题时,先查询文档再尝试

**下一步**：
- 开发策略：参考 [factor-examples.md](factor-examples.md)
- 数据处理：参考 [dataframe-reference.md](dataframe-reference.md)
- 因子分析：参考 [factor-analysis-reference.md](factor-analysis-reference.md)
