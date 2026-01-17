# 数据处理工具库

## 核心原则

**不要试图记住所有 pandas 方法，而是学会如何查询和扩展工具函数。**

本文档提供：
1. 核心工具函数（10-15个）
2. 工具函数设计模式
3. 如何扩展工具函数
4. pandas 高级用法查询指南

---

## 概述

在 A股量化策略开发中，数据处理是最频繁的操作。本文档提供一套核心工具函数，帮助你高效处理多股票时间序列数据（面板数据）。

**数据结构约定**：
- 索引（index）：日期（datetime）
- 列（columns）：股票代码（如 '600000.SH'）
- 值（values）：指标数值

```python
import pandas as pd

# 标准数据结构示例
#             600000.SH  000001.SZ  600519.SH
# 2023-01-03      10.5       15.2       1850.0
# 2023-01-04      10.8       15.5       1880.0
# 2023-01-05      10.6       15.3       1865.0
```

---

## 核心工具函数

### 1. 移动平均（average）

计算 n 期移动平均。如果窗口内超过一半的值为 NaN，则返回 NaN。

**函数签名**：
```python
def average(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    计算移动平均

    参数:
        df: 数据 DataFrame（索引为日期，列为股票代码）
        n: 移动平均周期

    返回:
        移动平均 DataFrame
    """
    return df.rolling(window=n, min_periods=n//2+1).mean()
```

**使用示例**：
```python
import pandas as pd
import tushare as ts

# 获取数据
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取收盘价数据
close_data = []
for code in ['600000.SH', '000001.SZ']:
    df = pro.daily(ts_code=code, start_date='20230101', end_date='20231231')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    close_data.append(df[['close']].rename(columns={'close': code}))

close = pd.concat(close_data, axis=1)

# 计算移动平均
sma10 = average(close, 10)
sma60 = average(close, 60)

# 策略条件：价格突破60日均线
signal = close > sma60
```

---

### 2. 信号检测 - 入场（entry）

识别条件从 False 变为 True 的入场信号点。

**函数签名thon
def entry(condition: pd.DataFrame) -> pd.DataFrame:
    """
    识别入场信号（False -> True）

    参数:
        condition: 布尔型 DataFrame

    返回:
        入场信号 DataFrame（True 表示入场点）
    """
    # 当前为 True 且前一期为 False
    return condition & (~condition.shift(1).fillna(False))
```

**使用示例**：
```python
# 条件：价格在前10名
top10 = close.rank(axis=1, ascending=False) <= 10

# 找出进入前10名的时间点
entry_signals = entry(top10)

print("入场信号数量:", entry_signals.sum().sum())
```

---

### 3. 信号检测 - 出场（exit_signal）

识别条件从 True 变为 False 的出场信号点。

**函数签名**：
```python
def exit_signal(condition: pd.DataFrame) -> pd.DataFrame:
    """
    识别出场信号（True -> False）

    参数:
        condition: 布尔型 DataFrame

    返回:
        出场信号 DataFrame（True 表示出场点）
    """
    # 当前为 False 且前一期为 True
    return (~condition) & condition.shift(1).fillna(False)
```

**使用示例**：
```python
# 条件：价格在前10名
top10 = close.rank(axis=1, ascending=False) <= 10

# 找出跌出前10名的时间点
exit_signals = exit_signal(top10)

print("出场信号数量:", exit_signals.sum().sum())
```

---

### 4. 持有直到（hold_until）

从入场信号开始持有，直到出场信号出现。

**函数签名**：
```python
def hold_until(entry_cond: pd.DataFrame, exit_cond: pd.DataFrame) -> pd.DataFrame:
    """
    从入场信号持有到出场信号

    参数:
        entry_cond: 入场条件 DataFrame
        exit_cond: 出场条件 DataFrame

    返回:
        持仓 DataFrame（True 表示持有）
    """
    position = pd.DataFrame(False, index=entry_cond.index, columns=entry_cond.columns)

    for col in entry_cond.columns:
        holding = False
        for idx in entry_cond.index:
            if entry_cond.loc[idx, col]:
                holding = True
            elif exit_cond.loc[idx, col]:
                holding = False
            position.loc[idx, col] = holding

    return position
```

**使用示例**：
```python
# 入场：价格突破20日高点
entry_cond = close > close.rolling(20).max().shift(1)

# 出场：价格跌破60日均线
exit_cond = close < average(close, 60)

# 生成持仓信号
position = hold_until(entry_cond, exit_cond)
```

---

### 5. 选择最大值（is_largest）

返回每个日期横截面上最大的 n 个值。

**函数签名**：
```python
def is_largest(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    选择每日最大的 n 个值

    参数:
        df: 数据 DataFrame
        n: 选择数量

    返回:
        布尔型 DataFrame（True 表示被选中）
    """
    return df.rank(axis=1, ascending=False) <= n
```

**使用示例**：
```python
# 获取 ROE 数据
roe = pro.fina_indicator(ts_code='', start_date='20230101', end_date='20231231',
                         fields='ts_code,end_date,roe')

# 选择 ROE 最高的10只股票
top_roe = is_largest(roe, 10)
```

---

### 6. 选择最小值（is_smallest）

返回每个日期横截面上最小的 n 个值。

**函数签名**：
```python
def is_smallest(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    选择每日最小的 n 个值

    参数:
        df: 数据 DataFrame
        n: 选择数量

    返回:
        布尔型 DataFrame（True 表示被选中）
    """
    return df.rank(axis=1, ascending=True) <= n
```

**使用示例**：
```python
# 获取市净率数据
pb = pro.daily_basic(ts_code='', start_date='20230101', end_date='20231231',
                     fields='ts_code,trade_date,pb')

# 选择市净率最低的10只股票（价值策略）
low_pb = is_smallest(pb, 10)
```

---

### 7. 行业内排名（rank_by_industry）

计算每只股票在其所属行业内的百分位排名（0-1之间）。

**函数签名**：
```python
def rank_by_industry(df: pd.DataFrame, industry_map: dict) -> pd.DataFrame:
    """
    行业内排名

    参数:
        df: 数据 DataFrame
        industry_map: 股票代码到行业的映射字典 {'600000.SH': '银行', ...}

    返回:
        行业内排名 DataFrame（0=行业最低，1=行业最高）
    """
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    # 按行业分组
    industry_groups = {}
    for code, industry in industry_map.items():
        if code in df.columns:
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(code)

    # 对每个行业内的股票进行排名
    for industry, codes in industry_groups.items():
        industry_data = df[codes]
        industry_rank = industry_data.rank(axis=1, pct=True)
        result[codes] = industry_rank

    return result
```

**使用示例**：
```python
# 获取行业信息
stock_basic = pro.stock_basic(exchange='', list_status='L',
                              fields='ts_code,name,industry')
industry_map = dict(zip(stock_basic['ts_code'], stock_basic['industry']))

# 获取市盈率数据
pe = pro.daily_basic(ts_code='', start_date='20230101', end_date='20231231',
                     fields='ts_code,trade_date,pe')

# 行业内市盈率排名
pe_industry_rank = rank_by_industry(pe, industry_map)

# 选择行业内市盈率最低的30%
cheap_in_industry = pe_industry_rank < 0.3
```

---

### 8. 行业中性化（neutralize_industry）

去除行业效应，返回相对于行业均值的偏离值。

**函数签名**：
```python
def neutralize_industry(df: pd.DataFrame, industry_map: dict) -> pd.DataFrame:
    """
    行业中性化（去除行业效应）

    参数:
        df: 数据 DataFrame
        industry_map: 股票代码到行业的映射字典

    返回:
        行业中性化后的 DataFrame（值为相对行业均值的偏离）
    """
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    # 按行业分组
    industry_groups = {}
    for code, industry in industry_map.items():
        if code in df.columns:
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(code)

    # 对每个行业减去行业均值
    for industry, codes in industry_groups.items():
        industry_data = df[codes]
        industry_mean = industry_data.mean(axis=1)
        result[codes] = industry_data.sub(industry_mean, axis=0)

    return result
```

**使用示例**：
```python
# 市盈率行业中性化
pe_neutral = neutralize_industry(pe, industry_map)

# 正值表示相对行业贵，负值表示相对行业便宜
cheap_vs_industry = pe_neutral < 0
```

---

### 9. 财务数据日期对齐（align_financial_data）

将季度财务数据对齐到日线数据，使用前向填充避免未来函数。

**函数签名**：
```python
def align_financial_data(financial_df: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    将财务数据对齐到日线数据

    参数:
        financial_df: 财务数据 DataFrame（索引为财报日期）
        daily_index: 日线数据的日期索引

    返回:
        对齐后的 DataFrame
    """
    # 重新索引并前向填充
    aligned = financial_df.reindex(daily_index, method='ffill')
    return aligned
```

**使用示例**：
```python
# 获取季度财务数据
income = pro.income(ts_code='600000.SH', start_date='20220101', end_date='20231231',
                    fields='ts_code,end_date,revenue,n_income')
income['end_date'] = pd.to_datetime(income['end_date'])
income = income.set_index('end_date').sort_index()

# 获取日线数据
daily = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
daily['trade_date'] = pd.to_datetime(daily['trade_date'])
daily = daily.set_index('trade_date').sort_index()

# 对齐财务数据到日线
income_daily = align_financial_data(income, daily.index)

# 现在可以安全地使用财务数据进行策略计算
```

---

### 10. 上涨检测（is_rising）

判断当前值是否高于 n 期前的值。

**函数签名**：
```python
def is_rising(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """
    判断是否上涨

    参数:
        df: 数据 DataFrame
        n: 比较周期（默认1，即与前一期比较）

    返回:
        布尔型 DataFrame（True 表示上涨）
    """
    return df > df.shift(n)
```

**使用示例**：
```python
# 价格高于10天前
rising_10d = is_rising(close, 10)

# 连续3天上涨
consecutive_rise = is_rising(close, 1).rolling(3).sum() == 3
```

---

### 11. 下跌检测（is_falling）

判断当前值是否低于 n 期前的值。

**函数签名**：
```python
def is_falling(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """
    判断是否下跌

    参数:
        df: 数据 DataFrame
        n: 比较周期

    返回:
        布尔型 DataFrame（True 表示下跌）
    """
    return df < df.shift(n)
```

**使用示例**：
```python
# 价格低于20天前（下跌趋势）
falling_20d = is_falling(close, 20)

# 避免下跌趋势的股票
avoid = falling_20d
```

---

### 12. 条件持续（sustain）

检查条件在移动窗口内是否持续满足。

**函数签名**：
```python
def sustain(condition: pd.DataFrame, nwindow: int, nsatisfy: int = None) -> pd.DataFrame:
    """
    检查条件持续性

    参数:
        condition: 布尔型 DataFrame
        nwindow: 窗口长度
        nsatisfy: 窗口内需要满足的最小次数（默认等于 nwindow）

    返回:
        布尔型 DataFrame（True 表示条件持续满足）
    """
    if nsatisfy is None:
        nsatisfy = nwindow

    # 计算窗口内 True 的数量
    count = condition.rolling(window=nwindow).sum()
    return count >= nsatisfy
```

**使用示例**：
```python
# 连续3天上涨
rising_3days = sustain(is_rising(close, 1), nwindow=3, nsatisfy=3)

# 5天内至少4天上涨
rising_4of5 = sustain(is_rising(close, 1), nwindow=5, nsatisfy=4)
```

---

### 13. 横截面排名（cross_sectional_rank）

计算每个日期横截面上的百分位排名。

**函数签名**：
```python
def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    横截面排名（每日对所有股票排名）

    参数:
        df: 数据 DataFrame

    返回:
        排名 DataFrame（0=最低，1=最高）
    """
    return df.rank(axis=1, pct=True)
```

**使用示例**：
```python
# 市净率横截面排名
pb_rank = cross_sectional_rank(pb)

# 选择市净率最低的30%
value_stocks = pb_rank < 0.3
```

---

### 14. 标准化（standardize）

对每个日期横截面进行标准化（均值0，标准差1）。

**函数签名**：
```python
def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    横截面标准化

    参数:
        df: 数据 DataFrame

    返回:
        标准化后的 DataFrame
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return df.sub(mean, axis=0).div(std, axis=0)
```

**使用示例**：
```python
# 标准化市盈率
pe_std = standardize(pe)

# 选择市盈率低于均值1个标准差的股票
cheap = pe_std < -1
```

---

### 15. 去极值（winsorize）

将极端值限制在指定百分位范围内。

**函数签名**：
```python
def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    去极值处理

    参数:
        df: 数据 DataFrame
        lower: 下限百分位（默认1%）
        upper: 上限百分位（默认99%）

    返回:
        去极值后的 DataFrame
    """
    result = df.copy()

    for date in df.index:
        row = df.loc[date]
        lower_bound = row.quantile(lower)
        upper_bound = row.quantile(upper)
        result.loc[date] = row.clip(lower=lower_bound, upper=upper_bound)

    return result
```

**使用示例**：
```python
# 去除市盈率极端值
pe_winsorized = winsorize(pe, lower=0.05, upper=0.95)

# 避免极端值影响策略
```

---

## 工具函数设计模式

### 模式1：时间序列处理

处理单只股票的时间序列数据（沿着时间轴）。

**特点**：
- 操作沿着索引（时间）方向
- 使用 `shift()`, `rolling()`, `expanding()` 等方法
- 注意避免未来函数

**示例**：
```python
def moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """移动平均 - 时间序列处理"""
    return df.rolling(window=window).mean()

def momentum(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """动量 - 时间序列处理"""
    return df / df.shift(period) - 1
```

---

### 模式2：横截面处理

处理某个时间点上所有股票的数据（沿着股票轴）。

**特点**：
- 操作沿着列（股票）方向
- 使用 `axis=1` 参数
- 适合选股和排名

**示例**：
```python
def select_top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """选择前 n 名 - 横截面处理"""
    return df.rank(axis=1, ascending=False) <= n

def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """横截面 Z-score - 横截面处理"""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return df.sub(mean, axis=0).div(std, axis=0)
```

---

### 模式3：面板数据处理

同时处理时间和股票两个维度的数据。

**特点**：
- 需要同时考虑时间和横截面
- 通常先时间序列处理，再横截面处理
- 注意数据对齐

**示例**：
```python
def relative_strength(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """相对强度 - 面板数据处理"""
    # 步骤1：计算每只股票的收益率（时间序列）
    returns = df / df.shift(period) - 1

    # 步骤2：横截面排名
    rank = returns.rank(axis=1, pct=True)

    return rank
```

---

## 如何扩展工具函数

### 步骤1：确定需求

明确你需要什么功能：
- 时间序列处理？横截面处理？面板数据处理？
- 输入是什么？输出是什么？
- 是否需要避免未来函数？

### 步骤2：查询 pandas 文档

使用 Context7 查询 pandas 官方文档：

```
Library ID: /pandas-dev/pandas

查询模板：
"pandas [方法名] [功能描述]"

示例：
- "pandas rolling 移动窗口"
- "pandas rank 排名 axis"
- "pandas groupby 分组"
- "pandas shift 滞后"
```

### 步骤3：参考核心工具函数

参考本文档中的核心工具函数实现模式：
- 函数签名设计
- 参数命名规范
- 返回值类型
- 注释风格

### 步骤4：编写和测试

```python
def your_function(df: pd.DataFrame, param: int) -> pd.DataFrame:
    """
    功能描述

    参数:
        df: 数据 DataFrame
        param: 参数说明

    返回:
        结果 DataFrame
    """
    # 实现代码
    result = df.rolling(param).mean()  # 示例
    return result

# 测试
test_data = pd.DataFrame({
    '600000.SH': [10, 11, 12, 13, 14],
    '000001.SZ': [20, 21, 22, 23, 24]
}, index=pd.date_range('2023-01-01', periods=5))

result = your_function(test_data, 3)
print(result)
```

---

## pandas 高级用法查询指南

### 如何使用 Context7 查询 pandas 文档

```
Library ID: /pandas-dev/pandas

常用查询关键词：
- "pandas DataFrame 方法"
- "pandas rolling 移动窗口"
- "pandas groupby 分组聚合"
- "pandas merge 合并"
- "pandas pivot 透视表"
- "pandas apply 自定义函数"
```

### 常用操作速查表

| 操作类型 | pandas 方法 | 查询关键词 |
|---------|------------|-----------|
| 移动窗口 | rolling() | "pandas rolling" |
| 滞后/超前 | shift() | "pandas shift" |
| 排名 | rank() | "pandas rank" |
| 分组 | groupby() | "pandas groupby" |
| 透视 | pivot_table() | "pandas pivot" |
| 合并 | merge(), concat() | "pandas merge concat" |
| 重采样 | resample() | "pandas resample" |
| 填充缺失值 | fillna() | "pandas fillna" |

---

## 完整示例：组合使用工具函数

```python
import pandas as pd
import tushare as ts

# 初始化
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取数据
stocks = ['600000.SH', '000001.SZ', '600519.SH']
close_data = []

for code in stocks:
    df = pro.daily(ts_code=code, start_date='20230101', end_date='20231231')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    close_data.append(df[['close']].rename(columns={'close': code}))

close = pd.concat(close_data, axis=1)

# 策略开发：组合使用工具函数

# 1. 计算技术指标
sma20 = average(close, 20)
sma60 = average(close, 60)

# 2. 生成条件
cond1 = close > sma60  # 价格在60日均线上方
cond2 = is_rising(close, 5)  # 5日上涨
cond3 = sustain(is_rising(close, 1), 3, 3)  # 连续3日上涨

# 3. 组合条件
entry_cond = cond1 & cond2 & cond3

# 4. 选股
selected = is_largest(close, 10) & entry_cond

# 5. 生成持仓信号
exit_cond = close < sma20
position = hold_until(selected, exit_cond)

print("持仓信号生成完成")
print(f"总信号数: {position.sum().sum()}")
```

---

## 注意事项

### 1. 避免未来函数

**错误示例**：
```python
# 错误：使用了未来数据
future_max = close.max()  # 使用了整个时间序列的最大值
signal = close > future_max * 0.9
```

**正确示例**：
```python
# 正确：只使用历史数据
rolling_max = close.rolling(window=20).max()
signal = close > rolling_max * 0.9
```

### 2. 数据对齐

不同频率的数据需要对齐：
```python
# 日线数据
daily_close = ...

# 季度财务数据
quarterly_roe = ...

# 对齐到日线
roe_daily = align_financial_data(quarterly_roe, daily_close.index)
```

### 3. 缺失值处理

```python
# 检查缺失值
print(df.isnull().sum())

# 前向填充（常用于财务数据）
df_filled = df.fillna(method='ffill')

# 删除缺失值
df_clean = df.dropna()
```

### 4. 性能优化

```python
# 避免循环，使用向量化操作
# 错误：慢
result = pd.DataFrame()
for col in df.columns:
    result[col] = df[col].rolling(20).mean()

# 正确：快
result = df.rolling(20).mean()
```

---

## 相关文档

- [data-reference.md](data-reference.md) - 数据获取参考
- [factor-examples.md](factor-examples.md) - 策略示例
- [factor-analysis-reference.md](factor-analysis-reference.md) - 因子分析工具
- [best-practices.md](best-practices.md) - 最佳实践

---

## 总结

**核心原则**：
1. ✅ 不要试图记住所有 pandas 方法
2. ✅ 学会使用 Context7 查询 pandas 文档
3. ✅ 参考核心工具函数的设计模式
4. ✅ 遵循"时间序列 → 横截面 → 面板数据"的处理流程
5. ✅ 始终注意避免未来函数

**下一步**：
- 查看策略示例：[factor-examples.md](factor-examples.md)
- 学习因子分析：[factor-analysis-reference.md](factor-analysis-reference.md)
- 了解最佳实践：[best-practices.md](best-practices.md)
