# 因子分析工具

## 核心原则

**不要试图记住所有统计方法和分析技巧，而是学会如何查询和应用因子分析方法论。**

本文档提供：
1. 因子分析方法论和流程
2. 核心分析工具（6个）
3. 典型因子分析示例（4个）
4. 如何使用 Context7 查询统计方法

---

## 概述

因子分析是量化投资的核心环节，用于评估因子的有效性、计算因子贡献度、优化因子组合。本文档提供适配 A股市场和 Backtrader 框架的因子分析工具。

**与 FinLab 的区别**：
- FinLab 使用内置的因子分析模块
- AShare 使用 pandas + numpy + scipy 实现因子分析
- 数据结构：DataFrame（索引为日期，列为股票代码）

---

## 因子分析方法论

### 因子分析流程

```
1. 定义因子
   ↓
2. 计算因子值（使用 dataframe-reference.md 中的工具函数）
   ↓
3. 评估因子有效性（IC、因子收益、回归统计）
   ↓
4. 分析因子贡献度（Shapley 值、因子组合）
   ↓
5. 优化因子组合（多因子策略）
   ↓
6. 回测验证（使用 backtesting-reference.md）
```

### 如何评估因子有效性

**核心指标**：

1. **IC（信息系数）**：因子值与未来收益的相关性
   - IC > 0.05：因子有效
   - IC > 0.1：因子强有效
   - Rank IC 更稳健（使用排名而非原始值）

2. **因子收益**：持有因子选出的股票组合的收益
   - 正收益：因子有效
   - 夏普比率 > 1：因子质量高

3. **IC 趋势**：IC 是否随时间衰减
   - 上升趋势：因子越来越有效
   - 下降趋势：因子失效，需要调整

4. **因子贡献度**：多因子策略中各因子的边际贡献
   - Shapley 值：衡量因子的真实贡献
   - 正贡献：因子有价值
   - 负贡献：因子拖累策略

### 如何使用 Context7 查询统计方法

```
Library ID: /scipy/scipy 或 /numpy/numpy

查询模板：
"[方法名] [功能描述] [参数]"

示例：
- "pearsonr 相关系数 计算方法"
- "spearmanr 秩相关 使用示例"
- "linregress 线性回归 返回值"
- "ttest_ind t检验 显著性"
- "corrcoef 相关矩阵 numpy"
```

---

## 核心分析工具

### 1. IC（信息系数）计算

计算因子值与未来收益的相关性，评估因子预测能力。

**函数实现**：
```python
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def calc_ic(factor: pd.DataFrame,
            future_return: pd.DataFrame,
            method: str = 'spearman') -> pd.Series:
    """
    计算因子的 IC（信息系数）

    参数:
        factor: 因子值 DataFrame（索引为日期，列为股票代码）
        future_return: 未来收益 DataFrame（索引为日期，列为股票代码）
        method: 相关系数方法，'spearman'（秩相关，推荐）或 'pearson'（线性相关）

    返回:
        pd.Series: 每个日期的 IC 值（索引为日期）
    """
    # 对齐数据
    common_dates = factor.index.intersection(future_return.index)
    factor = factor.loc[common_dates]
    future_return = future_return.loc[common_dates]

    ic_list = []
    for date in common_dates:
        # 获取当日因子值和未来收益
        factor_values = factor.loc[date].dropna()
        return_values = future_return.loc[date].dropna()

        # 找到共同的股票
        common_stocks = factor_values.index.intersection(return_values.index)
        if len(common_stocks) < 10:  # 至少需要10只股票
            ic_list.append(np.nan)
            continue

        factor_values = factor_values.loc[common_stocks]
        return_values = return_values.loc[common_stocks]

        # 计算相关系数
        if method == 'spearman':
            ic, _ = spearmanr(factor_values, return_values)
        else:  # pearson
            ic = np.corrcoef(factor_values, return_values)[0, 1]

        ic_list.append(ic)

    return pd.Series(ic_list, index=common_dates, name='IC')

# 使用示例
# 假设已经有因子值和价格数据
# factor = ...  # 因子值 DataFrame
# price = ...   # 价格 DataFrame

# 计算未来10日收益
future_return = price.pct_change(10).shift(-10)

# 计算 IC
ic_series = calc_ic(factor, future_return, method='spearman')

# 分析 IC 统计
print(f"平均 IC: {ic_series.mean():.4f}")
print(f"IC 标准差: {ic_series.std():.4f}")
print(f"IC IR (信息比率): {ic_series.mean() / ic_series.std():.4f}")
print(f"IC > 0 的比例: {(ic_series > 0).sum() / len(ic_series):.2%}")

# 可视化 IC 时间序列
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
ic_series.plot()
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.title('Factor IC Over Time')
plt.ylabel('IC')
plt.grid(True)
plt.show()
```

**关键点**：
- 使用 Spearman 秩相关（Rank IC）更稳健，对异常值不敏感
- IC > 0.05 表示因子有效，IC > 0.1 表示因子强有效
- IC IR（信息比率）= 平均IC / IC标准差，衡量IC的稳定性
- IC > 0 的比例应该 > 50%

---

### 2. 因子收益计算

计算持有因子选出的股票组合的收益，直观评估因子效果。

**函数实现**：
```python
import pandas as pd
import numpy as np

def calc_factor_return(factor_signal: pd.DataFrame,
                       future_return: pd.DataFrame) -> pd.Series:
    """
    计算因子收益（等权组合收益）

    参数:
        factor_signal: 因子信号 DataFrame（True表示选中，False表示不选中）
        future_return: 未来收益 DataFrame（索引为日期，列为股票代码）

    返回:
        pd.Series: 每个日期的因子收益（索引为日期）
    """
    # 对齐数据
    common_dates = factor_signal.index.intersection(future_return.index)
    factor_signal = factor_signal.loc[common_dates]
    future_return = future_return.loc[common_dates]

    factor_returns = []
    for date in common_dates:
        # 获取当日选中的股票
        selected_stocks = factor_signal.loc[date]
        selected_stocks = selected_stocks[selected_stocks == True].index

        if len(selected_stocks) == 0:
            factor_returns.append(np.nan)
            continue

        # 计算等权组合收益
        returns = future_return.loc[date, selected_stocks]
        avg_return = returns.mean()
        factor_returns.append(avg_return)

    return pd.Series(factor_returns, index=common_dates, name='Factor_Return')

# 使用示例
# 假设已经有因子值和价格数据
# factor = ...  # 因子值 DataFrame
# price = ...   # 价格 DataFrame

# 生成因子信号（选择因子值最大的30%股票）
factor_signal = factor.rank(axis=1, pct=True) > 0.7

# 计算未来10日收益
future_return = price.pct_change(10).shift(-10)

# 计算因子收益
factor_return = calc_factor_return(factor_signal, future_return)

# 计算累计收益
cumulative_return = (1 + factor_return).cumprod()

# 分析因子收益统计
print(f"平均收益: {factor_return.mean():.4f}")
print(f"年化收益: {factor_return.mean() * 252 / 10:.2%}")  # 假设10日持有期
print(f"收益标准差: {factor_return.std():.4f}")
print(f"夏普比率: {factor_return.mean() / factor_return.std() * np.sqrt(252 / 10):.2f}")
print(f"最大回撤: {(cumulative_return / cumulative_return.cummax() - 1).min():.2%}")

# 可视化累计收益
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
cumulative_return.plot()
plt.title('Factor Cumulative Return')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()
```

**关键点**：
- 因子收益是最直观的因子评估指标
- 等权组合：每只股票权重相同
- 夏普比率 > 1 表示因子质量高
- 需要考虑交易成本和滑点

---

### 3. 回归统计分析

使用线性回归分析因子的趋势和显著性，判断因子是否随时间衰减。

**函数实现**：
```python
import pandas as pd
import numpy as np
from scipy.stats import linregress

def calc_regression_stats(ic_series: pd.Series,
                          p_value_threshold: float = 0.05) -> dict:
    """
    对 IC 时间序列进行线性回归分析

    参数:
        ic_series: IC 时间序列（索引为日期）
        p_value_threshold: p值阈值，用于判断趋势显著性

    返回:
        dict: 包含斜率、p值、R²、趋势分类等统计信息
    """
    # 准备数据
    ic_series = ic_series.dropna()
    if len(ic_series) < 10:
        return {'error': '数据点太少，无法进行回归分析'}

    # 将日期转换为数值（天数）
    x = np.arange(len(ic_series))
    y = ic_series.values

    # 线性回归
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # 计算尾部估计值（最后一个点的预测值）
    tail_estimate = slope * (len(x) - 1) + intercept

    # 趋势分类
    if p_value < p_value_threshold:
        if slope > 0:
            trend = '显著上升'
        else:
            trend = '显著下降'
    else:
        trend = '无显著趋势'

    return {
        'slope': slope,              # 斜率（每日IC变化）
        'p_value': p_value,          # p值（显著性）
        'r_squared': r_value ** 2,   # R²（拟合优度）
        'tail_estimate': tail_estimate,  # 尾部估计值
        'trend': trend,              # 趋势分类
        'mean_ic': ic_series.mean(), # 平均IC
        'std_ic': ic_series.std()    # IC标准差
    }

# 使用示例
# 假设已经计算了 IC 时间序列
# ic_series = calc_ic(factor, future_return)

# 回归统计分析
stats = calc_regression_stats(ic_series)

print("IC 回归统计分析:")
print(f"斜率: {stats['slope']:.6f} (每日IC变化)")
print(f"p值: {stats['p_value']:.4f}")
print(f"R²: {stats['r_squared']:.4f}")
print(f"趋势: {stats['trend']}")
print(f"平均IC: {stats['mean_ic']:.4f}")
print(f"IC标准差: {stats['std_ic']:.4f}")

# 可视化回归线
import matplotlib.pyplot as plt
x = np.arange(len(ic_series))
y_pred = stats['slope'] * x + (stats['mean_ic'] - stats['slope'] * len(x) / 2)

plt.figure(figsize=(12, 6))
plt.plot(ic_series.values, label='IC', alpha=0.7)
plt.plot(x, y_pred, 'r--', label=f'Trend (slope={stats["slope"]:.6f})')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.title(f'IC Trend Analysis ({stats["trend"]})')
plt.ylabel('IC')
plt.legend()
plt.grid(True)
plt.show()
```

**关键点**：
- 显著下降趋势：因子可能失效，需要调整或放弃
- 显著上升趋势：因子越来越有效，可以增加权重
- 无显著趋势：因子稳定，适合长期使用
- p值 < 0.05 表示趋势显著

---

### 4. 特征和标签生成

将因子值转换为机器学习可用的特征和标签，用于因子组合优化。

**函数实现**：
```python
import pandas as pd
import numpy as np

def generate_features_and_labels(factors: dict,
                                 price: pd.DataFrame,
                                 holding_period: int = 10,
                                 top_pct: float = 0.3) -> tuple:
    """
    生成机器学习特征和标签

    参数:
        factors: 因子字典 {因子名: 因子值DataFrame}
        price: 价格 DataFrame（索引为日期，列为股票代码）
        holding_period: 持有期（天数）
        top_pct: 选择比例（0-1之间）

    返回:
        tuple: (features_df, labels_series)
            - features_df: 特征DataFrame（MultiIndex: date, stock_id）
            - labels_series: 标签Series（MultiIndex: date, stock_id）
    """
    # 计算未来收益作为标签
    future_return = price.pct_change(holding_period).shift(-holding_period)

    # 找到所有因子的共同日期
    common_dates = None
    for factor_name, factor_df in factors.items():
        if common_dates is None:
            common_dates = factor_df.index
        else:
            common_dates = common_dates.intersection(factor_df.index)

    common_dates = common_dates.intersection(future_return.index)

    # 构建特征和标签
    features_list = []
    labels_list = []

    for date in common_dates:
        # 获取当日所有因子值
        factor_values = {}
        for factor_name, factor_df in factors.items():
            factor_values[factor_name] = factor_df.loc[date]

        # 找到共同的股票
        common_stocks = None
        for factor_name, values in factor_values.items():
            values = values.dropna()
            if common_stocks is None:
                common_stocks = values.index
            else:
                common_stocks = common_stocks.intersection(values.index)

        # 获取未来收益
        returns = future_return.loc[date, common_stocks].dropna()
        common_stocks = common_stocks.intersection(returns.index)

        if len(common_stocks) < 10:
            continue

        # 构建特征矩阵
        for stock in common_stocks:
            feature_row = {}
            for factor_name, values in factor_values.items():
                feature_row[factor_name] = values.loc[stock]

            features_list.append({
                'date': date,
                'stock_id': stock,
                **feature_row
            })

            labels_list.append({
                'date': date,
                'stock_id': stock,
                'return': returns.loc[stock]
            })

    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    features_df = features_df.set_index(['date', 'stock_id'])

    labels_df = pd.DataFrame(labels_list)
    labels_series = labels_df.set_index(['date', 'stock_id'])['return']

    return features_df, labels_series

# 使用示例
# 假设已经有多个因子
# factor1 = ...  # 因子1 DataFrame
# factor2 = ...  # 因子2 DataFrame
# price = ...    # 价格 DataFrame

# 生成特征和标签
factors = {
    'momentum': factor1,
    'value': factor2
}

features, labels = generate_features_and_labels(factors, price, holding_period=10)

print(f"特征形状: {features.shape}")
print(f"标签形状: {labels.shape}")
print("\n特征示例:")
print(features.head())
print("\n标签示例:")
print(labels.head())
```

**关键点**：
- 特征：因子值（可以是原始值或排名）
- 标签：未来收益（可以是连续值或分类标签）
- MultiIndex 结构：(date, stock_id) 便于分组分析
- 用于机器学习模型训练和因子组合优化

---

### 5. 因子组合分析

分析多个因子的组合效果，找到最优因子组合。

**函数实现**：
```python
import pandas as pd
import numpy as np
from itertools import combinations

def analyze_factor_combinations(factors: dict,
                                price: pd.DataFrame,
                                holding_period: int = 10,
                                top_pct: float = 0.3) -> pd.DataFrame:
    """
    分析因子组合效果

    参数:
        factors: 因子字典 {因子名: 因子值DataFrame}
        price: 价格 DataFrame
        holding_period: 持有期（天数）
        top_pct: 选择比例

    返回:
        pd.DataFrame: 各因子组合的统计指标
    """
    # 计算未来收益
    future_return = price.pct_change(holding_period).shift(-holding_period)

    results = []

    # 单因子分析
    for factor_name, factor_df in factors.items():
        # 生成信号
        signal = factor_df.rank(axis=1, pct=True) > (1 - top_pct)

        # 计算收益
        factor_return = calc_factor_return(signal, future_return)

        # 统计指标
        results.append({
            'combination': factor_name,
            'mean_return': factor_return.mean(),
            'std_return': factor_return.std(),
            'sharpe_ratio': factor_return.mean() / factor_return.std() * np.sqrt(252 / holding_period),
            'win_rate': (factor_return > 0).sum() / len(factor_return)
        })

    # 多因子组合分析（两两组合）
    factor_names = list(factors.keys())
    for combo in combinations(factor_names, 2):
        # 组合信号（AND逻辑）
        signal = factors[combo[0]].rank(axis=1, pct=True) > (1 - top_pct)
        for factor_name in combo[1:]:
            signal = signal & (factors[factor_name].rank(axis=1, pct=True) > (1 - top_pct))

        # 计算收益
        factor_return = calc_factor_return(signal, future_return)

        if len(factor_return.dropna()) < 10:
            continue

        # 统计指标
        results.append({
            'combination': ' & '.join(combo),
            'mean_return': factor_return.mean(),
            'std_return': factor_return.std(),
            'sharpe_ratio': factor_return.mean() / factor_return.std() * np.sqrt(252 / holding_period),
            'win_rate': (factor_return > 0).sum() / len(factor_return)
        })

    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)

    return results_df

# 使用示例
# 假设已经有多个因子
# factors = {
#     'momentum': factor1,
#     'value': factor2,
#     'quality': factor3
# }

# 分析因子组合
combo_results = analyze_factor_combinations(factors, price, holding_period=10)

print("因子组合分析结果:")
print(combo_results.to_string(index=False))

# 可视化
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
combo_results.plot(x='combination', y='sharpe_ratio', kind='bar', ax=ax)
plt.title('Factor Combination Sharpe Ratios')
plt.ylabel('Sharpe Ratio')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
```

**关键点**：
- 单因子 vs 多因子：比较单独使用和组合使用的效果
- AND逻辑：同时满足多个因子条件（更严格）
- OR逻辑：满足任一因子条件（更宽松）
- 最优组合：夏普比率最高的组合

---

### 6. Shapley 值因子贡献度

使用 Shapley 值计算每个因子对组合收益的边际贡献。

**函数实现**：
```python
import pandas as pd
import numpy as np
from itertools import combinations, chain

def calc_shapley_values(factors: dict,
                       price: pd.DataFrame,
                       holding_period: int = 10,
                       top_pct: float = 0.3) -> pd.Series:
    """
    计算因子的 Shapley 值（边际贡献度）

    参数:
        factors: 因子字典 {因子名: 因子值DataFrame}
        price: 价格 DataFrame
        holding_period: 持有期（天数）
        top_pct: 选择比例

    返回:
        pd.Series: 各因子的 Shapley 值
    """
    # 计算未来收益
    future_return = price.pct_change(holding_period).shift(-holding_period)

    factor_names = list(factors.keys())
    n_factors = len(factor_names)

    # 计算所有子集的收益
    subset_returns = {}

    # 空集收益为0
    subset_returns[frozenset()] = 0

    # 计算所有非空子集的收益
    for r in range(1, n_factors + 1):
        for combo in combinations(factor_names, r):
            # 组合信号
            signal = factors[combo[0]].rank(axis=1, pct=True) > (1 - top_pct)
            for factor_name in combo[1:]:
                signal = signal & (factors[factor_name].rank(axis=1, pct=True) > (1 - top_pct))

            # 计算收益
            factor_return = calc_factor_return(signal, future_return)
            avg_return = factor_return.mean()

            subset_returns[frozenset(combo)] = avg_return

    # 计算 Shapley 值
    shapley_values = {}

    for factor in factor_names:
        shapley_value = 0

        # 遍历所有不包含该因子的子集
        for r in range(n_factors):
            other_factors = [f for f in factor_names if f != factor]
            for combo in combinations(other_factors, r):
                subset_without = frozenset(combo)
                subset_with = frozenset(list(combo) + [factor])

                # 边际贡献
                marginal_contribution = subset_returns[subset_with] - subset_returns[subset_without]

                # 权重：C(n-1, |S|) / C(n, |S|+1)
                weight = 1 / n_factors

                shapley_value += weight * marginal_contribution

        shapley_values[factor] = shapley_value

    return pd.Series(shapley_values)

# 使用示例
# 假设已经有多个因子
# factors = {
#     'momentum': factor1,
#     'value': factor2,
#     'quality': factor3
# }

# 计算 Shapley 值
shapley = calc_shapley_values(factors, price, holding_period=10)

print("因子 Shapley 值（边际贡献度）:")
print(shapley.sort_values(ascending=False))

# 可视化
import matplotlib.pyplot as plt
shapley.sort_values().plot(kind='barh', figsize=(10, 6))
plt.title('Factor Shapley Values (Marginal Contributions)')
plt.xlabel('Shapley Value')
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()
```

**关键点**：
- Shapley 值：衡量因子的真实贡献（考虑所有可能的组合）
- 正值：因子有正贡献，应该保留
- 负值：因子拖累策略，应该剔除
- 计算复杂度：O(2^n)，因子数量不宜过多（< 10个）

---

## 典型因子分析示例

### 示例1：单因子 IC 分析

完整的单因子 IC 分析流程，评估动量因子的有效性。

```python
import pandas as pd
import numpy as np
import tushare as ts
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# 设置 Tushare Token
pro = ts.pro_api('your_token')

# 1. 获取数据
# 获取股票列表（排除 ST 股）
stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
stock_list = [code for code in stock_basic['ts_code'] if 'ST' not in stock_basic[stock_basic['ts_code']==code]['name'].values[0]]

# 获取日线数据
start_date = '20200101'
end_date = '20231231'

price_data = {}
for stock in stock_list[:100]:  # 示例：只取前100只股票
    df = pro.daily(ts_code=stock, start_date=start_date, end_date=end_date)
    if len(df) > 0:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date').sort_index()
        price_data[stock] = df['close']

# 转换为 DataFrame
price = pd.DataFrame(price_data)

# 2. 计算动量因子（20日收益率）
momentum_factor = price.pct_change(20)

# 3. 计算未来10日收益
future_return = price.pct_change(10).shift(-10)

# 4. 计算 IC
ic_series = calc_ic(momentum_factor, future_return, method='spearman')

# 5. 分析 IC 统计
print("=" * 50)
print("动量因子 IC 分析")
print("=" * 50)
print(f"平均 IC: {ic_series.mean():.4f}")
print(f"IC 标准差: {ic_series.std():.4f}")
print(f"IC IR: {ic_series.mean() / ic_series.std():.4f}")
print(f"IC > 0 的比例: {(ic_series > 0).sum() / len(ic_series):.2%}")
print(f"IC > 0.05 的比例: {(ic_series > 0.05).sum() / len(ic_series):.2%}")

# 6. IC 趋势分析
stats = calc_regression_stats(ic_series)
print(f"\nIC 趋势: {stats['trend']}")
print(f"斜率: {stats['slope']:.6f}")
print(f"p值: {stats['p_value']:.4f}")

# 7. 可视化
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# IC 时间序列
axes[0].plot(ic_series.index, ic_series.values, label='IC', alpha=0.7)
axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
axes[0].axhline(y=0.05, color='g', linestyle='--', alpha=0.3, label='IC=0.05')
axes[0].set_title('Momentum Factor IC Over Time')
axes[0].set_ylabel('IC')
axes[0].legend()
axes[0].grid(True)

# IC 分布
axes[1].hist(ic_series.dropna(), bins=50, alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[1].axvline(x=ic_series.mean(), color='g', linestyle='--', alpha=0.5, label=f'Mean={ic_series.mean():.4f}')
axes[1].set_title('IC Distribution')
axes[1].set_xlabel('IC')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('momentum_ic_analysis.png', dpi=300)
print("\n图表已保存: momentum_ic_analysis.png")
```

**分析结论**：
- 如果平均 IC > 0.05 且 IC > 0 的比例 > 60%，说明动量因子有效
- 如果 IC 趋势为"显著下降"，需要考虑因子失效风险
- 如果 IC IR > 1，说明因子稳定性好

---

### 示例2：多因子 Shapley 值分析

分析多个因子的边际贡献度，找出最有价值的因子。

```python
import pandas as pd
import numpy as np
import tushare as ts

# 假设已经获取了价格数据
# price = ...

# 1. 计算多个因子
# 动量因子
momentum = price.pct_change(20)

# 价值因子（需要获取财务数据）
# 这里简化为市值因子（价格 * 成交量的代理）
# 实际应该使用 P/E、P/B 等财务指标
value = -price  # 负号表示价格越低越好

# 质量因子（这里简化为价格波动率的倒数）
volatility = price.pct_change().rolling(20).std()
quality = -volatility  # 负号表示波动率越低越好

# 2. 构建因子字典
factors = {
    'momentum': momentum,
    'value': value,
    'quality': quality
}

# 3. 计算 Shapley 值
shapley = calc_shapley_values(factors, price, holding_period=10, top_pct=0.3)

# 4. 分析结果
print("=" * 50)
print("多因子 Shapley 值分析")
print("=" * 50)
print("\n各因子边际贡献度:")
print(shapley.sort_values(ascending=False))

# 5. 因子组合分析
combo_results = analyze_factor_combinations(factors, price, holding_period=10, top_pct=0.3)
print("\n因子组合效果:")
print(combo_results.to_string(index=False))

# 6. 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Shapley 值
shapley.sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Factor Shapley Values')
axes[0].set_xlabel('Shapley Value (Marginal Contribution)')
axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[0].grid(True, axis='x')

# 因子组合夏普比率
combo_results.plot(x='combination', y='sharpe_ratio', kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Factor Combination Sharpe Ratios')
axes[1].set_ylabel('Sharpe Ratio')
axes[1].set_xticklabels(combo_results['combination'], rotation=45, ha='right')
axes[1].axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Sharpe=1')
axes[1].legend()
axes[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('multi_factor_analysis.png', dpi=300)
print("\n图表已保存: multi_factor_analysis.png")
```

**分析结论**：
- Shapley 值为正的因子应该保留
- Shapley 值为负的因子应该剔除
- 最优组合通常不是所有因子的简单组合

---

### 示例3：因子收益归因分析

分析因子收益的来源和稳定性。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 假设已经有因子和价格数据
# factor = ...  # 因子值 DataFrame
# price = ...   # 价格 DataFrame

# 1. 生成因子信号
factor_signal = factor.rank(axis=1, pct=True) > 0.7  # 选择前30%

# 2. 计算因子收益（不同持有期）
holding_periods = [5, 10, 20, 40, 60]
factor_returns = {}

for period in holding_periods:
    future_return = price.pct_change(period).shift(-period)
    factor_return = calc_factor_return(factor_signal, future_return)
    factor_returns[f'{period}D'] = factor_return

# 转换为 DataFrame
factor_returns_df = pd.DataFrame(factor_returns)

# 3. 计算统计指标
print("=" * 50)
print("因子收益归因分析")
print("=" * 50)
print("\n不同持有期的因子收益:")
print(factor_returns_df.describe())

# 4. 计算年化收益和夏普比率
print("\n年化统计:")
for period in holding_periods:
    col = f'{period}D'
    mean_return = factor_returns_df[col].mean()
    std_return = factor_returns_df[col].std()
    annual_return = mean_return * 252 / period
    annual_std = std_return * np.sqrt(252 / period)
    sharpe = annual_return / annual_std if annual_std > 0 else 0

    print(f"{col}:")
    print(f"  年化收益: {annual_return:.2%}")
    print(f"  年化波动: {annual_std:.2%}")
    print(f"  夏普比率: {sharpe:.2f}")

# 5. 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 累计收益
cumulative_returns = (1 + factor_returns_df).cumprod()
cumulative_returns.plot(ax=axes[0, 0])
axes[0, 0].set_title('Cumulative Returns by Holding Period')
axes[0, 0].set_ylabel('Cumulative Return')
axes[0, 0].legend(title='Holding Period')
axes[0, 0].grid(True)

# 收益分布
factor_returns_df.plot(kind='box', ax=axes[0, 1])
axes[0, 1].set_title('Return Distribution by Holding Period')
axes[0, 1].set_ylabel('Return')
axes[0, 1].grid(True)

# 滚动夏普比率（以10日持有期为例）
rolling_sharpe = factor_returns_df['10D'].rolling(60).mean() / factor_returns_df['10D'].rolling(60).std() * np.sqrt(252 / 10)
rolling_sharpe.plot(ax=axes[1, 0])
axes[1, 0].set_title('Rolling Sharpe Ratio (10D Holding Period, 60D Window)')
axes[1, 0].set_ylabel('Sharpe Ratio')
axes[1, 0].axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Sharpe=1')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 回撤分析
cumulative_10d = (1 + factor_returns_df['10D']).cumprod()
drawdown = cumulative_10d / cumulative_10d.cummax() - 1
drawdown.plot(ax=axes[1, 1], color='red')
axes[1, 1].set_title('Drawdown (10D Holding Period)')
axes[1, 1].set_ylabel('Drawdown')
axes[1, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('factor_return_attribution.png', dpi=300)
print("\n图表已保存: factor_return_attribution.png")
```

**分析结论**：
- 不同持有期的收益特征不同
- 最优持有期：夏普比率最高的持有期
- 滚动夏普比率：评估因子稳定性
- 回撤分析：评估因子风险

---

### 示例4：完整因子研究流程

从因子定义到回测验证的完整流程。

```python
import pandas as pd
import numpy as np
import tushare as ts
import backtrader as bt
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# 1. 定义因子
print("步骤1: 定义因子")
print("因子名称: 营收增长动量因子")
print("因子逻辑: 选择营收增速加快且价格动量强的股票")

# 2. 获取数据
print("\n步骤2: 获取数据")
# ... (数据获取代码，参考 data-reference.md)

# 3. 计算因子值
print("\n步骤3: 计算因子值")
# 营收增长率
revenue_growth = revenue.pct_change(4)  # 季度同比增长

# 价格动量
price_momentum = price.pct_change(20)

# 组合因子（标准化后相加）
from scipy.stats import zscore
revenue_growth_z = revenue_growth.apply(lambda x: zscore(x.dropna()), axis=1)
price_momentum_z = price_momentum.apply(lambda x: zscore(x.dropna()), axis=1)
combined_factor = revenue_growth_z + price_momentum_z

# 4. 评估因子有效性
print("\n步骤4: 评估因子有效性")

# 4.1 IC 分析
future_return = price.pct_change(10).shift(-10)
ic_series = calc_ic(combined_factor, future_return, method='spearman')

print(f"平均 IC: {ic_series.mean():.4f}")
print(f"IC IR: {ic_series.mean() / ic_series.std():.4f}")
print(f"IC > 0 比例: {(ic_series > 0).sum() / len(ic_series):.2%}")

# 4.2 IC 趋势分析
stats = calc_regression_stats(ic_series)
print(f"IC 趋势: {stats['trend']}")

# 4.3 因子收益分析
factor_signal = combined_factor.rank(axis=1, pct=True) > 0.7
factor_return = calc_factor_return(factor_signal, future_return)

print(f"平均收益: {factor_return.mean():.4f}")
print(f"夏普比率: {factor_return.mean() / factor_return.std() * np.sqrt(252 / 10):.2f}")

# 5. 因子优化
print("\n步骤5: 因子优化")
# 测试不同的选股比例
top_pcts = [0.1, 0.2, 0.3, 0.4, 0.5]
optimization_results = []

for pct in top_pcts:
    signal = combined_factor.rank(axis=1, pct=True) > (1 - pct)
    ret = calc_factor_return(signal, future_return)
    sharpe = ret.mean() / ret.std() * np.sqrt(252 / 10)

    optimization_results.append({
        'top_pct': pct,
        'mean_return': ret.mean(),
        'sharpe_ratio': sharpe
    })

opt_df = pd.DataFrame(optimization_results)
print(opt_df.to_string(index=False))

best_pct = opt_df.loc[opt_df['sharpe_ratio'].idxmax(), 'top_pct']
print(f"\n最优选股比例: {best_pct:.0%}")

# 6. 回测验证
print("\n步骤6: 回测验证")
print("使用 Backtrader 进行完整回测...")
# ..backtesting-reference.md)

# 7. 生成报告
print("\n步骤7: 生成因子研究报告")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# IC 时间序列
axes[0, 0].plot(ic_series.index, ic_series.values)
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
axes[0, 0].set_title('Factor IC Over Time')
axes[0, 0].set_ylabel('IC')
axes[0, 0].grid(True)

# 因子收益累计曲线
cumulative_return = (1 + factor_return).cumprod()
axes[0, 1].plot(cumulative_return.index, cumulative_return.values)
axes[0, 1].set_title('Factor Cumulative Return')
axes[0, 1].set_ylabel('Cumulative Return')
axes[0, 1].grid(True)

# 选股比例优化
axes[1, 0].plot(opt_df['top_pct'], opt_df['sharpe_ratio'], marker='o')
axes[1, 0].axvline(x=best_pct, color='r', linestyle='--', alpha=0.5, label=f'Best={best_pct:.0%}')
axes[1, 0].set_title('Sharpe Ratio by Stock Selection Ratio')
axes[1, 0].set_xlabel('Top %')
axes[1, 0].set_ylabel('Sharpe Ratio')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 因子值分布
axes[1, 1].hist(combined_factor.iloc[-1].dropna(), bins=50, alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Factor Value Distribution (Latest)')
axes[1, 1].set_xlabel('Factor Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('complete_factor_research.png', dpi=300)
print("\n因子研究报告已保存: complete_factor_research.png")
```

**研究结论模板**：
1. **因子定义**: [因子名称和逻辑]
2. **因子有效性**: IC = [值], IC IR = [值], 趋势 = [上升/下降/稳定]
3. **因子收益**: 年化收益 = [值], 夏普比率 = [值]
4. **最优参数**: 选股比例 = [值], 持有期 = [值]
5. **风险评估**: 最大回撤 = [值], 胜率 = [值]
6. **建议**: [保留/优化/放弃]

---

## 如何进行因子研究

### 因子研究检查清单

**1. 因子定义阶段**
- [ ] 因子逻辑清晰，有经济学或行为金融学解释
- [ ] 因子计算方法明确，可复现
- [ ] 考虑了 A股市场特点（T+1、涨跌停等）

**2. 数据准备阶段**
- [ ] 数据来源可靠（Tushare Pro）
- [ ] 数据质量检查（缺失值、异常值）
- [ ] 数据对齐（日期、股票代码）
- [ ] 排除 ST 股和停牌股

**3. 因子评估阶段**
- [ ] 计算 IC 和 IC IR
- [ ] 分析 IC 趋势（是否衰减）
- [ ] 计算因子收益和夏普比率
- [ ] 检查因子稳定性（滚动窗口分析）

**4. 因子优化阶段**
- [ ] 测试不同选股比例
- [ ] 测试不同持有期
- [ ] 测试因子组合
- [ ] 行业中性化（如需要）

**5. 回测验证阶段**
- [ ] 使用 Backtrader 进行完整回测
- [ ] 考虑交易成本和滑点
- [ ] 分析回撤和风险指标
- [ ] 样本外验证

**6. 风险控制阶段**
- [ ] 检查前视偏差
- [ ] 检查数据泄露
- [ ] 检查过拟合
- [ ] 压力测试（极端市场环境）

---

## 相关文档

- [数据处理工具库](dataframe-reference.md) - 因子计算所需的工具函数
- [数据参考](data-reference.md) - 获取因子所需的数据
- [回测参考](backtesting-reference.md) - 因子策略回测
- [机器学习集成](machine-learning-reference.md) - 使用 ML 进行因子研究
- [最佳实践](best-practices.md) - 避免常见错误

---

## 总结

因子分析是量化投资的核心技能。本文档提供了：

1. **6个核心工具**：IC计算、因子收益、回归统计、特征标签生成、因子组合分析、Shapley值
2. **4个完整示例**：单因子IC分析、多因子Shapley值分析、因子收益归因、完整研究流程
3. **方法论指导**：如何评估因子、如何优化因子、如何避免常见错误

**记住**：
- 不要穷举所有统计方法，而是学会使用 Context7 查询
- 不要过度拟合历史数据，而是注重因子的经济学逻辑
- 不要忽视交易成本和市场冲击，而是进行真实的回测验证

**下一步**：
- 使用本文档的工具进行因子研究
- 参考 `machine-learning-reference.md` 进行 ML 因子研究
- 参考 `backtesting-reference.md` 进行策略回测
