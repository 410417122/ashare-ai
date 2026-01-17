# Tushare → Backtrader 数据对接指南

本文档详细说明如何将 Tushare Pro 获取的数据正确转换为 Backtrader 可用的格式。

---

## 核心问题

Tushare 和 Backtrader 的数据格式存在以下差异：

| 差异点 | Tushare | Backtrader |
|--------|---------|------------|
| 日期格式 | 字符串 `'20230101'` | datetime 索引 |
| 数据排序 | 降序（最新在前） | 升序（最早在前） |
| 成交量列名 | `vol` | `volume` |
| 索引 | 无特定要求 | 必须是 datetime 索引 |

---

## 标准转换函数

```python
import tushare as ts
import backtrader as bt
import pandas as pd

def tushare_to_backtrader(ts_code, start_date, end_date, pro):
    """
    将 Tushare 日线数据转换为 Backtrader 数据源

    参数:
        ts_code: 股票代码，如 '600000.SH'
        start_date: 开始日期，如 '20230101'
        end_date: 结束日期，如 '20231231'
        pro: tushare pro 接口对象

    返回:
        bt.feeds.PandasData 对象
    """
    # 1. 获取日线数据
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    if df.empty:
        raise ValueError(f"未获取到 {ts_code} 的数据，请检查代码和日期范围")

    # 2. 日期格式转换：字符串 → datetime
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # 3. 设置日期为索引
    df = df.set_index('trade_date')

    # 4. 升序排列（Backtrader 要求）
    df = df.sort_index(ascending=True)

    # 5. 列名映射：vol → volume
    df = df.rename(columns={'vol': 'volume'})

    # 6. 选择 OHLCV 列
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # 7. 创建 Backtrader 数据源
    data = bt.feeds.PandasData(dataname=df)

    return data
```

---

## 转换步骤详解

### 步骤 1: 获取原始数据

```python
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
```

**Tushare daily 返回字段：**
- `ts_code`: 股票代码
- `trade_date`: 交易日期 (字符串格式)
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `pre_close`: 昨收价
- `change`: 涨跌额
- `pct_chg`: 涨跌幅
- `vol`: 成交量 (手)
- `amount`: 成交额 (千元)

### 步骤 2: 日期格式转换

```python
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date')
```

**为什么必须转换？**
- Backtrader 内部使用 datetime 进行时间序列处理
- 字符串格式无法正确排序和比较

### 步骤 3: 升序排列

```python
df = df.sort_index(ascending=True)
```

**为什么必须升序？**
- Backtrader 按时间顺序遍历数据
- 降序数据会导致策略逻辑错误

### 步骤 4: 列名映射

```python
df = df.rename(columns={'vol': 'volume'})
```

**Backtrader 标准列名：**
- `open`, `high`, `low`, `close`, `volume`
- 可选: `openinterest` (期货用)

### 步骤 5: 创建数据源

```python
data = bt.feeds.PandasData(dataname=df)
```

---

## 复权数据处理

A股回测通常需要使用复权数据，避免除权除息导致的价格跳空。

### 使用 pro_bar 获取复权数据

```python
def get_adjusted_data(ts_code, start_date, end_date, pro, adj='hfq'):
    """
    获取复权数据

    参数:
        adj: 复权类型
            - 'qfq': 前复权
            - 'hfq': 后复权 (推荐用于回测)
            - None: 不复权
    """
    # 使用 ts.pro_bar 获取复权数据
    df = ts.pro_bar(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        adj=adj,
        api=pro
    )

    if df is None or df.empty:
        raise ValueError(f"未获取到 {ts_code} 的复权数据")

    # 标准转换流程
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date')
    df = df.sort_index(ascending=True)
    df = df.rename(columns={'vol': 'volume'})
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return bt.feeds.PandasData(dataname=df)
```

### 前复权 vs 后复权

| 复权类型 | 说明 | 适用场景 |
|---------|------|---------|
| 前复权 (qfq) | 以最新价格为基准向前调整 | 查看当前价格走势 |
| 后复权 (hfq) | 以上市价格为基准向后调整 | **回测推荐** |
| 不复权 | 原始价格 | 查看实际成交价 |

**回测为什么用后复权？**
- 后复权价格连续，不会因除权产生跳空
- 收益率计算更准确
- 历史数据不会随时间变化

---

## 多股票数据加载

```python
def load_multiple_stocks(stock_list, start_date, end_date, pro):
    """
    加载多只股票数据到 Cerebro

    参数:
        stock_list: 股票代码列表，如 ['600000.SH', '000001.SZ']

    返回:
        cerebro 对象
    """
    cerebro = bt.Cerebro()

    for ts_code in stock_list:
        try:
            # 获取数据
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

            if df.empty:
                print(f"警告: {ts_code} 无数据，跳过")
                continue

            # 转换格式
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date')
            df = df.sort_index(ascending=True)
            df = df.rename(columns={'vol': 'volume'})
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # 添加到 cerebro，使用股票代码作为名称
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data, name=ts_code)

        except Exception as e:
            print(f"加载 {ts_code} 失败: {e}")
            continue

    return cerebro
```

---

## 数据对齐问题

多股票回测时，不同股票的交易日可能不同（停牌、新股等）。

### 解决方案：填充缺失数据

```python
def align_stock_data(df_dict, method='ffill'):
    """
    对齐多只股票的数据

    参数:
        df_dict: {股票代码: DataFrame} 字典
        method: 填充方法
            - 'ffill': 向前填充（用前一天数据）
            - 'drop': 删除任一股票缺失的日期

    返回:
        对齐后的 DataFrame 字典
    """
    # 获取所有交易日的并集
    all_dates = set()
    for df in df_dict.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    aligned_dict = {}
    for code, df in df_dict.items():
        # 重新索引到所有交易日
        df_aligned = df.reindex(all_dates)

        if method == 'ffill':
            # 向前填充缺失值
            df_aligned = df_aligned.fillna(method='ffill')

        aligned_dict[code] = df_aligned

    if method == 'drop':
        # 删除任一股票缺失的日期
        valid_dates = all_dates.copy()
        for df in aligned_dict.values():
            valid_dates = [d for d in valid_dates if not df.loc[d].isna().any()]

        for code in aligned_dict:
            aligned_dict[code] = aligned_dict[code].loc[valid_dates]

    return aligned_dict
```

---

## 常见错误及解决

### 错误 1: 数据为空

```
ValueError: 未获取到数据
```

**原因：**
- 股票代码格式错误（应为 `600000.SH` 而非 `600000`）
- 日期范围内无交易数据
- Tushare 积分不足

**解决：**
```python
# 检查股票代码格式
if not ts_code.endswith(('.SH', '.SZ')):
    raise ValueError("股票代码格式应为 XXXXXX.SH 或 XXXXXX.SZ")

# 先查询股票是否存在
stock_info = pro.stock_basic(ts_code=ts_code)
if stock_info.empty:
    raise ValueError(f"股票 {ts_code} 不存在")
```

### 错误 2: 数据类型错误

```
TypeError: Cannot compare type 'Timestamp' with type 'str'
```

**原因：** 日期未转换为 datetime

**解决：**
```python
df['trade_date'] = pd.to_datetime(df['trade_date'])
```

### 错误 3: 数据顺序错误

策略行为异常，买卖信号不正确

**原因：** 数据未按升序排列

**解决：**
```python
df = df.sort_index(ascending=True)
```

---

## 完整示例

```python
import tushare as ts
import backtrader as bt
import pandas as pd

# 1. 初始化 Tushare
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的中转Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 2. 获取并转换数据
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()
df = df.rename(columns={'vol': 'volume'})
df = df[['open', 'high', 'low', 'close', 'volume']]

# 3. 创建 Cerebro 并添加数据
cerebro = bt.Cerebro()
data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)

# 4. 设置初始资金
cerebro.broker.setcash(100000)

# 5. 运行
print(f'初始资金: {cerebro.broker.getvalue():.2f}')
results = cerebro.run()
print(f'最终资金: {cerebro.broker.getvalue():.2f}')
```

---

## 数据转换方法论

### 核心原则

1. **固定顺序**：始终按照相同的顺序转换数据（日期转换 → 设置索引 → 升序排列 → 列名映射 → 选择列）
2. **验证优先**：每一步都验证结果是否符合预期
3. **错误处理**：检查空数据、格式错误等异常情况
4. **文档查询**：遇到不确定的接口或参数，立即查询官方文档

### 转换流程图

```
Tushare 原始数据
    ↓
检查是否为空
    ↓
日期格式转换 (str → datetime)
    ↓
设置日期索引
    ↓
升序排列 (关键！)
    ↓
列名映射 (vol → volume)
    ↓
选择 OHLCV 列
    ↓
Backtrader 数据源
```

### 常见错误排查

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| 策略逻辑错误 | 数据未升序排列 | `df.sort_index(ascending=True)` |
| 成交量指标无效 | 列名不匹配 | `df.rename(columns={'vol': 'volume'})` |
| 时间序列操作失败 | 索引不是 datetime | `pd.to_datetime()` 转换 |
| 数据为空报错 | 未检查空数据 | 添加 `if df.empty` 检查 |

---

## 如何查询不确定的内容

遇到不确定的接口、参数或数据格式时，使用 Context7 查询官方文档：

### Tushare 数据接口

```
Library ID: /websites/tushare_pro_document
查询示例:
- "daily 接口返回字段说明"
- "pro_bar 复权参数 adj"
- "daily 和 pro_bar 的区别"
- "如何获取多只股票数据"
```

### Backtrader 数据格式

```
LibraID: /websites/backtrader_docu
查询示例:
- "PandasData 数据格式要求"
- "如何添加自定义数据列"
- "数据索引格式要求"
- "多股票数据加载方法"
```

### pandas 数据处理

```
查询示例:
- "pandas to_datetime 日期转换"
- "pandas sort_index 排序方法"
- "pandas reindex 重新索引参数"
- "pandas fillna 填充方法"
```

### 查询方法

1. **明确问题**：确定需要查询什么（接口用法、参数含义、返回格式）
2. **使用 Context7**：提供 Library ID 和具体查询关键词
3. **验证结果**：根据文档编写代码并测试
4. **记录经验**：将常用转换模式记录下来

**原则**：宁可多查一次文档，也不要凭猜测编写代码。

---

## 参考文档

- [data-reference.md](data-reference.md) - Tushare 数据参考和查询方法
- [best-practices.md](best-practices.md) - 数据处理最佳实践
- [debugging-guide.md](debugging-guide.md) - 数据验证方法
- [known-issues.md](known-issues.md) - 数据相关已知问题
