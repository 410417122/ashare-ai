# 已知问题和陷阱

本文档记录使用 Tushare Pro + Backtrader 进行 A 股量化交易时的已知问题和常见陷阱。

---

## 数据相关问题

### 1. 北向资金数据不可用 (重要)

**问题描述：**
`moneyflow_hsgt` (沪深港通资金流向) 接口从 2024 年起不再提供数据。

**影响：**
- 无法获取北向资金（外资）流入流出数据
- 依赖北向资金的策略无法实现

**解决方案：**
- 不要使用 `moneyflow_hsgt` 接口
- 如需外资数据，考虑其他数据源

```python
# ❌ 不要使用
df = pro.moneyflow_hsgt(start_date='20240101', end_date='20240131')

# ✅ 使用个股资金流向替代（部分场景）
df = pro.moneyflow(ts_code='000001.SZ', start_date='20240101', end_date='20240131')
```

---

### 2. Tushare 数据排序问题

**问题描述：**
Tushare 返回的数据默认是**降序**排列（最新日期在前），而 Backtrader 要求**升序**排列。

**影响：**
- 策略逻辑错误
- 指标计算错误
- 买卖信号颠倒

**解决方案：**
```python
# 必须升序排列
df = df.sort_index(ascending=True)
```

---

### 3. 成交量列名不一致

**问题描述：**
- Tushare 使用 `vol` 表示成交量
- Backtrader 使用 `volume` 表示成交量

**影响：**
- Backtrader 无法识别成交量数据
- 依赖成交量的指标无法计算

**解决方案：**
```python
df = df.rename(columns={'vol': 'volume'})
```

---

### 4. 日期格式问题

**问题描述：**
Tushare 返回的日期是字符串格式 `'20230101'`，Backtrader 需要 datetime 索引。

**影响：**
- 数据无法正确加载
- 时间序列操作失败

**解决方案：**
```python
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date')
```

---

### 5. 复权数据获取方式

**问题描述：**
`pro.daily()` 返回的是不复权数据，复权数据需要使用 `ts.pro_bar()`。

**影响：**
- 除权除息导致价格跳空
- 收益率计算不准确

**解决方案：**
```python
# 获取后复权数据
df = ts.pro_bar(ts_code='600000.SH', adj='hfq', start_date='20230101', end_date='20231231', api=pro)
```

---

## 接口限制问题

### 6. Tushare 积分限制

**问题描述：**
部分高级接口需要较高的 Tushare 积分才能使用。

**常见受限接口：**
- `stk_factor_pro` (每日指标)
- `fina_indicator` (财务指标)
- 部分行业数据接口

**解决方案：**
- 查询 context7 确认接口的积分要求
- 使用中转接口可能有不同的限制
- 联系中转服务提供商确认可用接口

---

### 7. 调用频率限制

**问题描述：**
Tushare 和中转接口都有调用频率限制。

**影响：**
- 批量获取数据时可能被限流
- 报错 `too many requests`

**解决方案：**
```python
import time

def get_data_with_retry(ts_code, start_date, end_date, pro, max_retries=3):
    for i in range(max_retries):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            if 'too many' in str(e).lower():
                time.sleep(1)  # 等待 1 秒后重试
            else:
                raise
    raise Exception(f"获取 {ts_code} 数据失败，已重试 {max_retries} 次")
```

---

## 回测相关问题

### 8. T+1 规则未处理

**问题描述：**
Backtrader 默认不处理 T+1 规则，当日买入当日可卖出。

**影响：**
- 回测结果过于乐观
- 与实际交易不符

**解决方案：**
参考 [ashare-rules.md](ashare-rules.md) 中的 T+1 策略基类实现。

---

### 9. 涨跌停未处理

**问题描述：**
Backtrader 默认不考虑涨跌停限制。

**影响：**
- 涨停时仍可买入（实际无法买入）
- 跌停时仍可卖出（实际无法卖出）
- 回测结果失真

**解决方案：**
参考 [ashare-rules.md](ashare-rules.md) 中的涨跌停处理逻辑。

---

### 10. 手续费计算不准确

**问题描述：**
Backtrader 默认的手续费设置不符合 A 股规则（印花税仅卖出收取）。

**影响：**
- 交易成本计算不准确
- 影响策略收益评估

**解决方案：**
使用自定义的 `AShareCommission` 类，参考 [SKILL.md](SKILL.md)。

---

### 11. 整手交易未处理

**问题描述：**
Backtrader 默认可以买入任意数量的股票，但 A 股要求买入必须是 100 股的整数倍。

**影响：**
- 回测中可能出现买入 50 股等不合规情况

**解决方案：**
使用自定义 Sizer，参考 [ashare-rules.md](ashare-rules.md)。

---

## 数据质量问题

### 12. 停牌数据缺失

**问题描述：**
停牌期间没有交易数据，DataFrame 中会缺少这些日期。

**影响：**
- 多股票回测时数据不对齐
- 可能导致索引错误

**解决方案：**
```python
# 方案1: 向前填充
df = df.reindex(all_trading_dates).fillna(method='ffill')

# 方案2: 跳过停牌股票
if df.empty or len(df) < min_days:
    continue
```

---

### 13. 新股数据异常

**问题描述：**
新股上市初期价格波动大，且有特殊的涨跌停规则。

**影响：**
- 策略在新股上表现异常
- 可能产生虚假信号

**解决方案：**
```python
# 过滤上市不足 60 天的新股
from datetime import datetime, timedelta

def is_new_stock(list_date, trade_date, min_days=60):
    list_dt = datetime.strptime(list_date, '%Y%m%d')
    trade_dt = datetime.strptime(trade_date, '%Y%m%d')
    return (trade_dt - list_dt).days < min_days
```

---

### 14. ST 股票数据

**问题描述：**
ST 股票涨跌停幅度为 5%，且可能随时退市。

**影响：**
- 策略参数可能不适用于 ST 股票
- 退市风险

**解决方案：**
建议在选股时过滤掉 ST 股票，参考 [ashare-rules.md](ashare-rules.md)。

---

## 代码常见错误

### 15. 股票代码格式错误

**问题描述：**
Tushare 要求股票代码带后缀（.SH 或 .SZ）。

```python
# ❌ 错误
df = pro.daily(ts_code='600000')

# ✅ 正确
df = pro.daily(ts_code='600000.SH')
```

---

### 16. 日期格式错误

**问题描述：**
Tushare 要求日期格式为 `YYYYMMDD`，不带分隔符。

```python
# ❌ 错误
df = pro.daily(ts_code='600000.SH', start_date='2023-01-01')

# ✅ 正确
df = pro.daily(ts_code='600000.SH', start_date='20230101')
```

---

### 17. 空数据未检查

**问题描述：**
接口可能返回空 DataFrame，未检查会导致后续代码报错。

```python
# ❌ 危险
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
df = df.set_index('trade_date')  # 如果 df 为空会报错

# ✅ 安全
df = pro.daily(ts_code='600000.SH', start_date='20230101', end_date='20231231')
if df.empty:
    raise ValueError("未获取到数据")
df = df.set_index('trade_date')
```

---

### 18. 默认买入数量为 1 股

**问题描述：**
Backtrader 默认 Sizer 买入 1 股，而不是根据资金计算。

**影响：**
- 10 万资金只买 1 股，收益几乎为零
- 回测结果完全失真

**解决方案：**
```python
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

cerebro.addsizer(AShareSizer)
```

---

### 19. CommInfoBase 的 percabs 参数

**问题描述：**
自定义 `CommInfoBase` 时，`commission` 参数的解释取决于 `percabs` 参数：
- `percabs=False`（默认）：`0.1` 表示 0.1%
- `percabs=True`：`0.001` 表示 0.1%

**影响：**
- 传入 `commission=0.0003` 但 `percabs=False`，实际佣金是 0.0003% 而非 0.03%
- 手续费计算严重偏低

**解决方案：**
```python
class AShareCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.0003),
        ('percabs', True),  # 关键：必须设置为 True
        # ...
    )
```

---

## 问题排查清单

遇到问题时，按以下顺序检查：

1. **数据是否为空？** `print(len(df))`
2. **数据是否升序？** `print(df.index[:5])`
3. **列名是否正确？** `print(df.columns.tolist())`
4. **日期格式是否正确？** `print(type(df.index[0]))`
5. **股票代码格式是否正确？** 应为 `XXXXXX.SH` 或 `XXXXXX.SZ`
6. **日期格式是否正确？** 应为 `YYYYMMDD`

---

## 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-01-17 | 初始版本，记录已知问题 |

---

## 发现新问题？

如果在使用过程中发现新的问题或陷阱，请更新此文档。
