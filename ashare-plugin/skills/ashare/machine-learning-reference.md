# 机器学习集成

## 核心原则

**不要试图穷举所有机器学习方法和参数，而是学会如何查询和应用 ML 工作流程。**

本文档提供：
1. ML 工作流程和方法论
2. 特征工程模板
3. ML 框架集成示例（2-3个）
4. 如何使用 Context7 查询 ML 库文档

---

## 概述

机器学习在量化投资中的应用越来越广泛。本文档提供适配 A股市场和 Backtrader 框架的 ML 集成方案，帮助你使用机器学习开发量化策略。

**与 FinLab 的区别**：
- FinLab 提供内置的 `finlab.ml` 模块
- AShare 使用 pandas + scikit-learn + XGBoost 等标准 ML 库
- 数据结构：DataFrame（索引为日期，列为股票代码）

**适用场景**：
- 多因子策略优化
- 股票收益预测
- 风险预测和控制
- 特征选择和降维

---

## ML 工作流程

### 完整工作流程

```
1. 数据准备
   ↓
2. 特征工程（使用 dataframe-reference.md 中的工具函数）
   ↓
3. 标签生成（使用 factor-analysis-reference.md 中的方法）
   ↓
4. 数据清洗和对齐
   ↓
5. 训练集/测试集划分（时间序列划分）
   ↓
6. 模型训练
   ↓
7. 模型评估和调优
   ↓
8. 生成交易信号
   ↓
9. 回测验证（使用 backtesting-reference.md）
   ↓
10. 实盘部署
```

### 数据准备阶段

**关键步骤**：
1. 获取价格数据（日线、周线、月线）
2. 获取基本面数据（财务指标、估值指标）
3. 获取技术指标数据
4. 数据对齐和清洗

**示例代码**：
```python
import pandas as pd
import tushare as ts

# 初始化 Tushare
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取股票列表（排除 ST 股）
stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
stock_list = [code for code in stock_basic['ts_code'][:100]
              if 'ST' not in stock_basic[stock_basic['ts_code']==code]['name'].values[0]]

# 获取价格数据
start_date = '20200101'
end_date = '20231231'

price_data = {}
for stock in stock_list:
    df = pro.daily(ts_code=stock, start_date=start_date, end_date=end_date)
    if len(df) > 0:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date').sort_index()
        price_data[stock] = df['close']

# 转换为 DataFrame
price = pd.DataFrame(price_data)
print(f"价格数据形状: {price.shape}")
```

### 特征工程阶段

**特征类型**：
1. **技术指标特征**：MACD、RSI、KD、布林带等
2. **基本面特征**：P/E、P/B、ROE、营收增长等
3. **筹码特征**：北向资金、融资融券、机构持仓等
4. **衍生特征**：动量、波动率、相对强度等

**特征工程原则**：
- 避免未来函数（不使用未来数据）
- 特征标准化（均值0，标准差1）
- 处理缺失值（前向填充或删除）
- 特征选择（去除冗余特征）

**参考文档**：
- 使用 `dataframe-reference.md` 中的工具函数计算特征
- 使用 `factor-analysis-reference.md` 中的方法评估特征有效性

### 标签生成阶段

**标签类型**：
1. **回归标签**：未来收益率（连续值）
2. **分类标签**：涨跌方向（0/1）
3. **排名标签**：相对排名（0-1之间）

**标签生成原则**：
- 标签必须是未来数据（避免数据泄露）
- 标签周期与交易频率匹配
- 考虑交易成本和滑点

**参考文档**：
- 使用 `factor-analysis-reference.md` 中的 `generate_features_and_labels` 函数

### 模型训练阶段

**模型选择**：
1. **线性模型**：Ridge、Lasso、ElasticNet（可解释性强）
2. **树模型**：RandomForest、XGBoost、LightGBM（非线性关系）
3. **神经网络**：MLP、LSTM（复杂模式）

**训练原则**：
- 使用时间序列划分（不使用随机划分）
- 交叉验证（时间序列交叉验证）
- 超参数调优（网格搜索或贝叶斯优化）
- 防止过拟合（正则化、早停）

### 模型评估阶段

**评估指标**：
1. **回归任务**：R²、MSE、MAE、IC
2. **分类任务**：准确率、精确率、召回率、AUC
3. **排名任务**：Rank IC、NDCG

**评估原则**：
- 样本外测试（测试集）
- 时间序列交叉验证
- 稳定性分析（滚动窗口）

### 如何使用 Context7 查询 ML 库文档

```
常用 ML 库的 Library ID：

1. scikit-learn: /scikit-learn/scikit-learn
   查询示例：
   - "RandomForestRegressor 参数说明"
   - "train_test_split 时间序列"
   - "StandardScaler 使用方法"
   - "GridSearchCV 交叉验证"

2. XGBoost: /dmlc/xgboost
   查询示例：
   - "XGBRegressor 参数调优"
   - "early_stopping_rounds 使用"
   - "feature_importances_ 特征重要性"

3. pandas: /pandas-dev/pandas
   查询示例：
   - "MultiIndex 多层索引"
   - "pivot_table 透视表"
   - "rolling 移动窗口"

4. numpy: /numpy/numpy
   查询示例：
   - "corrcoef 相关系数"
   - "percentile 百分位数"
```

---

## 特征工程模板

### 技术指标特征生成模板

```python
import pandas as pd
import numpy as np

def generate_technical_features(price: pd.DataFrame) -> dict:
    """
    生成技术指标特征

    参数:
        price: 价格 DataFrame（索引为日期，列为股票代码）

    返回:
        dict: 特征字典 {特征名: 特征DataFrame}
    """
    features = {}

    # 1. 动量特征
    features['momentum_5d'] = price.pct_change(5)
    features['momentum_10d'] = price.pct_change(10)
    features['momentum_20d'] = price.pct_change(20)

    # 2. 移动平均特征
    features['sma_10'] = price.rolling(10).mean()
    features['sma_20'] = price.rolling(20).mean()
    features['sma_60'] = price.rolling(60).mean()

    # 3. 价格相对位置
    features['price_to_sma20'] = price / features['sma_20']
    features['price_to_sma60'] = price / features['sma_60']

    # 4. 波动率特征
    returns = price.pct_change()
    features['volatility_10d'] = returns.rolling(10).std()
    features['volatility_20d'] = returns.rolling(20).std()

    # 5. 相对强度特征（横截面排名）
    features['momentum_rank'] = features['momentum_20d'].rank(axis=1, pct=True)

    return features

# 使用示例
# price = ...  # 价格 DataFrame
# tech_features = generate_technical_features(price)
# print(f"生成了 {len(tech_features)} 个技术指标特征")
```

**扩展方法**：
- 使用 Context7 查询 TA-Lib 库获取更多技术指标
- 参考 `dataframe-reference.md` 中的工具函数
- 根据策略需求自定义技术指标

### 基本面特征生成模板

```python
import pandas as pd
import tushare as ts

def generate_fundamental_features(stock_list: list,
                                  start_date: str,
                                  end_date: str,
                                  pro: ts.pro_api) -> dict:
    """
    生成基本面特征

    参数:
        stock_list: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        pro: Tushare Pro API 对象

    返回:
        dict: 特征字典 {特征名: 特征DataFrame}
    """
    features = {}

    # 1. 估值指标
    # 获取每日估值数据
    pe_data = {}
    pb_data = {}
    ps_data = {}

    for stock in stock_list:
        df = pro.daily_basic(ts_code=stock, start_date=start_date, end_date=end_date,
                            fields='ts_code,trade_date,pe,pb,ps')
        if len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date').sort_index()
            pe_data[stock] = df['pe']
            pb_data[stock] = df['pb']
            ps_data[stock] = df['ps']

    features['pe'] = pd.DataFrame(pe_data)
    features['pb'] = pd.DataFrame(pb_data)
    features['ps'] = pd.DataFrame(ps_data)

    # 2. 估值相对位置（横截面排名）
    features['pe_rank'] = features['pe'].rank(axis=1, pct=True)
    features['pb_rank'] = features['pb'].rank(axis=1, pct=True)

    # 3. 估值变化（时间序列）
    features['pe_change'] = features['pe'].pct_change(20)
    features['pb_change'] = features['pb'].pct_change(20)

    return features

# 使用示例
# pro = ts.pro_api('your_token')
# stock_list = ['600000.SH', '000001.SZ']
# fund_features = generate_fundamental_features(stock_list, '20200101', '20231231', pro)
```

**扩展方法**：
- 参考 `data-reference.md` 获取更多财务数据
- 使用 Context7 查询 Tushare 文档获取更多指标
- 计算财务指标的增长率和趋势

### 标签生成方法

```python
import pandas as pd
import numpy as np

def generate_regression_label(price: pd.DataFrame,
                              holding_period: int = 10) -> pd.DataFrame:
    """
    生成回归标签（未来收益率）

    参数:
        price: 价格 DataFrame
        holding_period: 持有期（天数）

    返回:
        pd.DataFrame: 未来收益率
    """
    # 计算未来收益率
    future_return = price.pct_change(holding_period).shift(-holding_period)
    return future_return


def generate_classification_label(price: pd.DataFrame,
                                  holding_period: int = 10,
                                  threshold: float = 0.0) -> pd.DataFrame:
    """
    生成分类标签（涨跌方向）

    参数:
        price: 价格 DataFrame
        holding_period: 持有期（天数）
        threshold: 涨跌阈值（默认0，即涨为1，跌为0）

    返回:
        pd.DataFrame: 分类标签（0或1）
    """
    # 计算未来收益率
    future_return = price.pct_change(holding_period).shift(-holding_period)

    # 转换为分类标签
    label = (future_return > threshold).astype(int)
    return label


def generate_excess_return_label(price: pd.DataFrame,
                                 holding_period: int = 10) -> pd.DataFrame:
    """
    生成超额收益标签（相对市场平均）

    参数:
        price: 价格 DataFrame
        holding_period: 持有期（天数）

    返回:
        pd.DataFrame: 超额收益
    """
    # 计算未来收益率
    future_return = price.pct_change(holding_period).shift(-holding_period)

    # 计算市场平均收益
    market_return = future_return.mean(axis=1)

    # 计算超额收益
    excess_return = future_return.sub(market_return, axis=0)
    return excess_return

# 使用示例
# price = ...  # 价格 DataFrame
#
# # 回归标签
# y_regression = generate_regression_label(price, holding_period=10)
#
# # 分类标签
# y_classification = generate_classification_label(price, holding_period=10)
#
# # 超额收益标签
# y_excess = generate_excess_return_label(price, holding_period=10)
```

**标签选择建议**：
- **回归标签**：适合预测具体收益率，可以用于排名选股
- **分类标签**：适合预测涨跌方向，简单直观
- **超额收益标签**：适合相对收益策略，去除市场整体波动

---

## ML 框架集成示例

### 示例1：scikit-learn 随机森林回归

完整的 ML 工作流程，使用随机森林预测股票收益。

```python
import pandas as pd
import numpy as np
import tushare as ts
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# ========== 1. 数据准备 ==========
print("步骤1: 数据准备")

# 初始化 Tushare
pro = ts.pro_api('占位符')
pro._DataApi__token = '你的Token'
pro._DataApi__http_url = 'http://tushare.xiximiao.com/dataapi'

# 获取股票列表
stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
stock_list = [code for code in stock_basic['ts_code'][:50]  # 示例：50只股票
              if 'ST' not in stock_basic[stock_basic['ts_code']==code]['name'].values[0]]

# 获取价格数据
start_date = '20200101'
end_date = '20231231'

price_data = {}
for stock in stock_list:
    df = pro.daily(ts_code=stock, start_date=start_date, end_date=end_date)
    if len(df) > 0:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date').sort_index()
        price_data[stock] = df['close']

price = pd.DataFrame(price_data)
print(f"价格数据形状: {price.shape}")

# ========== 2. 特征工程 ==========
print("\n步骤2: 特征工程")

# 生成技术指标特征
tech_features = generate_technical_features(price)

# 合并特征为 MultiIndex DataFrame
feature_list = []
for date in price.index:
    for stock in price.columns:
        row = {'date': date, 'stock': stock}
        for feat_name, feat_df in tech_features.items():
            if date in feat_df.index and stock in feat_df.columns:
                row[feat_name] = feat_df.loc[date, stock]
        feature_list.append(row)

features_df = pd.DataFrame(feature_list)
features_df = features_df.set_index(['date', 'stock'])

print(f"特征数据形状: {features_df.shape}")
print(f"特征列: {features_df.columns.tolist()}")

# ========== 3. 标签生成 ==========
print("\n步骤3: 标签生成")

# 生成未来10日收益率标签
holding_period = 10
future_return = generate_regression_label(price, holding_period)

# 转换为 Series（MultiIndex）
label_list = []
for date in future_return.index:
    for stock in future_return.columns:
        if not pd.isna(future_return.loc[date, stock]):
            label_list.append({
                'date': date,
                'stock': stock,
                'return': future_return.loc[date, stock]
            })

labels_df = pd.DataFrame(label_list)
labels_series = labels_df.set_index(['date', 'stock'])['return']

print(f"标签数据形状: {labels_series.shape}")

# ========== 4. 数据清洗和对齐 ==========
print("\n步骤4: 数据清洗")

# 对齐特征和标签
common_index = features_df.index.intersection(labels_series.index)
X = features_df.loc[common_index]
y = labels_series.loc[common_index]

# 删除包含 NaN 的行
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

print(f"清洗后数据形状: X={X.shape}, y={y.shape}")

# ========== 5. 训练集/测试集划分 ==========
print("\n步骤5: 数据划分")

# 时间序列划分（前80%训练，后20%测试）
dates = X.index.get_level_values('date').unique().sort_values()
split_date = dates[int(len(dates) * 0.8)]

train_mask = X.index.get_level_values('date') < split_date
test_mask = X.index.get_level_values('date') >= split_date

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"训练集: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"测试集: X_test={X_test.shape}, y_test={y_test.shape}")
print(f"划分日期: {split_date}")

# ========== 6. 特征标准化 ==========
print("\n步骤6: 特征标准化")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 7. 模型训练 ==========
print("\n步骤7: 模型训练")

model = RandomForestRegressor(
    n_estimators=100,      # 树的数量
    max_depth=10,          # 最大深度
    min_samples_split=20,  # 最小分裂样本数
    min_samples_leaf=10,   # 叶子节点最小样本数
    random_state=42,
    n_jobs=-1              # 使用所有CPU核心
)

model.fit(X_train_scaled, y_train)
print("模型训练完成")

# ========== 8. 模型评估 ==========
print("\n步骤8: 模型评估")

# 训练集评估
train_score = model.score(X_train_scaled, y_train)
print(f"训练集 R²: {train_score:.4f}")

# 测试集评估
test_score = model.score(X_test_scaled, y_test)
print(f"测试集 R²: {test_score:.4f}")

# 预测
y_pred = model.predict(X_test_scaled)

# 计算 IC（信息系数）
from scipy.stats import spearmanr
ic, _ = spearmanr(y_test, y_pred)
print(f"测试集 IC: {ic:.4f}")

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特征重要性（Top 5）:")
print(feature_importance.head())

# ========== 9. 生成交易信号 ==========
print("\n步骤9: 生成交易信号")

# 将预测结果转换为 DataFrame
pred_df = pd.DataFrame({
    'date': X_test.index.get_level_values('date'),
    'stock': X_test.index.get_level_values('stock'),
    'pred_return': y_pred
})

# 转换为宽格式（日期 x 股票）
pred_wide = pred_df.pivot(index='date', columns='stock', values='pred_return')

# 选择预测收益最高的10只股票
top_n = 10
position = pred_wide.rank(axis=1, ascending=False) <= top_n

print(f"交易信号形状: {position.shape}")
print(f"平均持仓数量: {position.sum(axis=1).mean():.1f}")

# ========== 10. 简单回测 ==========
print("\n步骤10: 简单回测")

# 计算实际收益
actual_return_wide = y_test.reset_index().pivot(index='date', columns='stock', values='return')

# 对齐持仓和收益
common_dates = position.index.intersection(actual_return_wide.index)
position_aligned = position.loc[common_dates]
return_aligned = actual_return_wide.loc[common_dates]

# 计算策略收益（等权）
strategy_return = (position_aligned * return_aligned).sum(axis=1) / position_aligned.sum(axis=1)

# 计算累计收益
cumulative_return = (1 + strategy_return).cumprod()

# 计算统计指标
total_return = cumulative_return.iloc[-1] - 1
annual_return = (1 + total_return) ** (252 / len(strategy_return)) - 1
sharpe_ratio = strategy_return.mean() / strategy_return.std() * np.sqrt(252)
max_drawdown = (cumulative_return / cumulative_return.cummax() - 1).min()

print(f"总收益: {total_return:.2%}")
print(f"年化收益: {annual_return:.2%}")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"最大回撤: {max_drawdown:.2%}")

# 可视化
plt.figure(figsize=(12, 6))
cumulative_return.plot()
plt.title('ML Strategy Cumulative Return')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.savefig('ml_strategy_return.png', dpi=300)
print("\n图表已保存: ml_strategy_return.png")
```

**关键点**：
- 使用时间序列划分（不使用随机划分）
- 特征标准化（StandardScaler）
- 防止过拟合（限制树深度、最小样本数）
- 评估 IC 和 R²
- 简单回测验证策略效果

---

### 示例2：XGBoost 梯度提升回归

使用 XGBoost 进行股票收益预测，包含特征选择和超参数调优。

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

# 假设已经完成了数据准备、特征工程、标签生成、数据清洗
# X_train, y_train, X_test, y_test 已经准备好

# ========== 1. 特征选择 ==========
print("步骤1: 特征选择")

# 使用 F-统计量选择最重要的特征
k_best = 8  # 选择8个最重要的特征
selector = SelectKBest(score_func=f_regression, k=k_best)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 获取选中的特征名
selected_features = X_train.columns[selector.get_support()].tolist()
print(f"选中的特征: {selected_features}")

# 转换为 DataFrame
X_train_selected = pd.DataFrame(X_train_selected, index=X_train.index, columns=selected_features)
X_test_selected = pd.DataFrame(X_test_selected, index=X_test.index, columns=selected_features)

# ========== 2. 特征标准化 ==========
print("\n步骤2: 特征标准化")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# ========== 3. XGBoost 模型训练 ==========
print("\n步骤3: XGBoost 模型训练")

# 创建 DMatrix（XGBoost 的数据结构）
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# 设置参数
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'max_depth': 6,                   # 树的最大深度
    'learning_rate': 0.05,            # 学习率
    'subsample': 0.8,                 # 样本采样比例
    'colsample_bytree': 0.8,          # 特征采样比例
    'min_child_weight': 5,            # 最小叶子节点权重
    'gamma': 0.1,                     # 最小分裂损失
    'reg_alpha': 0.1,                 # L1 正则化
    'reg_lambda': 1.0,                # L2 正则化
    'random_state': 42
}

# 训练模型（带早停）
evals = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,              # 最大迭代次数
    evals=evals,
    early_stopping_rounds=50,         # 早停轮数
    verbose_eval=50                   # 每50轮打印一次
)

print(f"\n最佳迭代次数: {model.best_iteration}")
print(f"最佳测试分数: {model.best_score:.4f}")

# ========== 4. 模型评估 ==========
print("\n步骤4: 模型评估")

# 预测
y_train_pred = model.predict(dtrain)
y_test_pred = model.predict(dtest)

# 计算 R²
from sklearn.metrics import r2_score
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"训练集 R²: {train_r2:.4f}")
print(f"测试集 R²: {test_r2:.4f}")

# 计算 IC
from scipy.stats import spearmanr
train_ic, _ = spearmanr(y_train, y_train_pred)
test_ic, _ = spearmanr(y_test, y_test_pred)

print(f"训练集 IC: {train_ic:.4f}")
print(f"测试集 IC: {test_ic:.4f}")

# ========== 5. 特征重要性分析 ==========
print("\n步骤5: 特征重要性分析")

# 获取特征重要性
importance_dict = model.get_score(importance_type='gain')  # 使用 gain 作为重要性指标

# 转换为 DataFrame
importance_df = pd.DataFrame({
    'feature': list(importance_dict.keys()),
    'importance': list(importance_dict.values())
}).sort_values('importance', ascending=False)

# 映射回特征名
feature_map = {f'f{i}': name for i, name in enumerate(selected_features)}
importance_df['feature_name'] = importance_df['feature'].map(feature_map)

print("\n特征重要性:")
print(importance_df[['feature_name', 'importance']])

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature_name'], importance_df['importance'])
plt.xlabel('Importance (Gain)')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png', dpi=300)
print("\n特征重要性图已保存: xgboost_feature_importance.png")

# ========== 6. 生成交易信号 ==========
print("\n步骤6: 生成交易信号")

# 将预测结果转换为 DataFrame
pred_df = pd.DataFrame({
    'date': X_test.index.get_level_values('date'),
    'stock': X_test.index.get_level_values('stock'),
    'pred_return': y_test_pred
})

# 转换为宽格式
pred_wide = pred_df.pivot(index='date', columns='stock', values='pred_return')

# 选择预测收益最高的10只股票
position = pred_wide.rank(axis=1, ascending=False) <= 10

print(f"交易信号生成完成，形状: {position.shape}")

# ========== 7. 回测验证 ==========
# （参考示例1的回测代码）
```

**关键点**：
- 特征选择（SelectKBest）减少特征数量
- XGBoost 参数调优（学习率、树深度、正则化）
- 早停机制（early_stopping_rounds）防止过拟合
- 特征重要性分析（gain）
- 使用 DMatrix 提高效率

**如何查询 XGBoost 文档**：
```
Library ID: /dmlc/xgboost

查询示例：
- "XGBoost 参数说明"
- "early_stopping_rounds 使用方法"
- "feature_importances 计算方式"
- "xgb.train vs XGBRegressor 区别"
```

---

### 示例3：特征选择和交叉验证

使用时间序列交叉验证评估模型稳定性。

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

# 假设已经完成了数据准备、特征工程、标签生成、数据清洗
# X, y 已经准备好（包含训练集和验证集）

# ========== 1. 时间序列交叉验证 ==========
print("步骤1: 时间序列交叉验证")

# 获取唯一日期
dates = X.index.get_level_values('date').unique().sort_values()
print(f"总日期数: {len(dates)}")

# 创建时间序列分割器（5折）
tscv = TimeSeriesSplit(n_splits=5)

# 存储每折的结果
fold_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(dates)):
    print(f"\n=== Fold {fold + 1} ===")

    # 获取训练和验证日期
    train_dates = dates[train_idx]
    val_dates = dates[val_idx]

    print(f"训练期: {train_dates[0]} 到 {train_dates[-1]}")
    print(f"验证期: {val_dates[0]} 到 {val_dates[-1]}")

    # 根据日期划分数据
    X_train_fold = X[X.index.get_level_values('date').isin(train_dates)]
    y_train_fold = y[X_train_fold.index]
    X_val_fold = X[X.index.get_level_values('date').isin(val_dates)]
    y_val_fold = y[X_val_fold.index]

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)

    # 训练模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train_fold)

    # 评估
    train_score = model.score(X_train_scaled, y_train_fold)
    val_score = model.score(X_val_scaled, y_val_fold)

    # 计算 IC
    from scipy.stats import spearmanr
    y_val_pred = model.predict(X_val_scaled)
    val_ic, _ = spearmanr(y_val_fold, y_val_pred)

    print(f"训练集 R²: {train_score:.4f}")
    print(f"验证集 R²: {val_score:.4f}")
    print(f"验证集 IC: {val_ic:.4f}")

    fold_results.append({
        'fold': fold + 1,
        'train_r2': train_score,
        'val_r2': val_score,
        'val_ic': val_ic
    })

# 汇总结果
results_df = pd.DataFrame(fold_results)
print("\n=== 交叉验证汇总 ===")
print(results_df)
print(f"\n平均验证 R²: {results_df['val_r2'].mean():.4f} (+/- {results_df['val_r2'].std():.4f})")
print(f"平均验证 IC: {results_df['val_ic'].mean():.4f} (+/- {results_df['val_ic'].std():.4f})")

# ========== 2. 递归特征消除（RFE）==========
print("\n步骤2: 递归特征消除")

# 使用全部数据进行特征选择
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建 RFECV（带交叉验证的递归特征消除）
model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)

# 注意：RFECV 不支持时间序列交叉验证，这里仅作演示
# 实际应用中应该手动实现时间序列版本的 RFE
rfecv = RFECV(
    estimator=model,
    step=1,                    # 每次消除1个特征
    cv=3,                      # 3折交叉验证
    scoring='r2',              # 评分指标
    n_jobs=-1
)

print("开始特征选择（可能需要较长时间）...")
rfecv.fit(X_scaled, y)

print(f"\n最优特征数量: {rfecv.n_features_}")
print(f"选中的特征: {X.columns[rfecv.support_].tolist()}")

# 可视化特征数量 vs 性能
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
         rfecv.cv_results_['mean_test_score'])
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Score (R²)')
plt.title('Recursive Feature Elimination with Cross-Validation')
plt.axvline(x=rfecv.n_features_, color='r', linestyle='--', label=f'Optimal: {rfecv.n_features_}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rfecv_results.png', dpi=300)
print("\n特征选择结果图已保存: rfecv_results.png")

# ========== 3. 使用选中的特征重新训练 ==========
print("\n步骤3: 使用选中的特征重新训练")

X_selected = X.iloc[:, rfecv.support_]
print(f"选中的特征: {X_selected.columns.tolist()}")

# 划分训练集和测试集
split_date = dates[int(len(dates) * 0.8)]
train_mask = X_selected.index.get_level_values('date') < split_date
test_mask = X_selected.index.get_level_values('date') >= split_date

X_train = X_selected[train_mask]
y_train = y[train_mask]
X_test = X_selected[test_mask]
y_test = y[test_mask]

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练最终模型
final_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train_scaled, y_train)

# 评估
test_score = final_model.score(X_test_scaled, y_test)
y_test_pred = final_model.predict(X_test_scaled)
test_ic, _ = spearmanr(y_test, y_test_pred)

print(f"最终模型测试集 R²: {test_score:.4f}")
print(f"最终模型测试集 IC: {test_ic:.4f}")
```

**关键点**：
- 时间序列交叉验证（TimeSeriesSplit）评估模型稳定性
- 递归特征消除（RFECV）自动选择最优特征数量
- 每折独立进行特征标准化
- 汇总多折结果评估模型泛化能力

**注意事项**：
- RFECV 默认使用 KFold，不适合时间序列数据
- 实际应用中应该手动实现时间序列版本的 RFE
- 特征选择过程可能耗时较长

---

## 如何开发 ML 策略

### ML 策略开发流程

```
1. 定义预测目标
   - 预测什么？（收益率、涨跌方向、排名）
   - 预测周期？（日、周、月）
   - 评估指标？（R²、IC、准确率）

2. 生成特征和标签
   - 技术指标特征
   - 基本面特征
   - 标签生成（避免未来函数）

3. 数据清洗和预处理
   - 处理缺失值
   - 特征标准化
   - 数据对齐

4. 特征选择
   - 相关性分析
   - 特征重要性
   - 递归特征消除

5. 模型选择和训练
   - 选择合适的模型
   - 超参数调优
   - 交叉验证

6. 模型评估
   - 样本外测试
   - IC 分析
   - 稳定性分析

7. 生成交易信号
   - 预测值转换为持仓
   - 风险控制
   - 仓位管理

8. 回测验证
   - 使用 Backtrader 完整回测
   - 考虑交易成本
   - 分析回撤和风险

9. 模型监控和更新
   - 监控模型性能
   - 定期重新训练
   - 模型版本管理
```

### ML 策略检查清单

**数据准备阶段**
- [ ] 数据来源可靠（Tushare Pro）
- [ ] 数据质量检查（缺失值、异常值）
- [ ] 数据对齐（日期、股票代码）
- [ ] 排除 ST 股和停牌股

**特征工程阶段**
- [ ] 避免未来函数（不使用未来数据）
- [ ] 特征有经济学或技术分析解释
- [ ] 特征标准化或归一化
- [ ] 处理缺失值（前向填充或删除）

**标签生成阶段**
- [ ] 标签是未来数据（避免数据泄露）
- [ ] 标签周期与交易频率匹配
- [ ] 考虑交易成本和滑点

**模型训练阶段**
- [ ] 使用时间序列划分（不使用随机划分）
- [ ] 交叉验证（时间序列交叉验证）
- [ ] 超参数调优（网格搜索或贝叶斯优化）
- [ ] 防止过拟合（正则化、早停、剪枝）

**模型评估阶段**
- [ ] 样本外测试（测试集）
- [ ] 评估 IC 和 R²
- [ ] 稳定性分析（滚动窗口）
- [ ] 特征重要性分析

**回测验证阶段**
- [ ] 使用 Backtrader 进行完整回测
- [ ] 考虑交易成本和滑点
- [ ] 分析回撤和风险指标
- [ ] 样本外验证

**风险控制阶段**
- [ ] 检查前视偏差
- [ ] 检查数据泄露
- [ ] 检查过拟合
- [ ] 压力测试（极端市场环境）

---

## 最佳实践

### 1. 避免常见错误

**错误1：使用未来数据**
```python
# 错误：使用了未来数据
future_max = price.max()  # 使用了整个时间序列的最大值
feature = price / future_max

# 正确：只使用历史数据
rolling_max = price.rolling(window=20).max()
feature = price / rolling_max
```

**错误2：随机划分训练集和测试集**
```python
# 错误：随机划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 正确：时间序列划分
split_date = dates[int(len(dates) * 0.8)]
train_mask = X.index.get_level_values('date') < split_date
test_mask = X.index.get_level_values('date') >= split_date
X_train = X[train_mask]
X_test = X[test_mask]
```

**错误3：在全部数据上进行特征标准化**
```python
# 错误：在全部数据上标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 使用了测试集的信息

# 正确：只在训练集上 fit
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 只 transform
```

**错误4：过度拟合**
```python
# 错误：模型过于复杂
model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,  # 无限深度
    min_samples_split=2  # 最小分裂样本数太小
)

# 正确：限制模型复杂度
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10
)
```

### 2. 特征工程技巧

**技巧1：特征交互**
```python
# 创建特征交互项
X['momentum_x_volatility'] = X['momentum_20d'] * X['volatility_20d']
X['pe_x_momentum'] = X['pe'] * X['momentum_20d']
```

**技巧2：特征分箱**
```python
# 将连续特征转换为分类特征
X['pe_bin'] = pd.cut(X['pe'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
X = pd.get_dummies(X, columns=['pe_bin'])
```

**技巧3：滞后特征**
```python
# 创建滞后特征
for lag in [1, 5, 10]:
    X[f'momentum_lag{lag}'] = X['momentum_20d'].shift(lag)
```

### 3. 模型调优技巧

**技巧1：网格搜索**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [10, 20, 30]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
print(f"最佳参数: {grid_search.best_params_}")
```

**技巧2：学习曲线分析**
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train,
    cv=3, scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# 可视化学习曲线
plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
plt.show()
```

### 4. 模型集成

**方法1：简单平均**
```python
# 训练多个模型
model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model2 = xgb.XGBRegressor(n_estimators=100, random_state=42)

model1.fit(X_train_scaled, y_train)
model2.fit(X_train_scaled, y_train)

# 预测并平均
y_pred1 = model1.predict(X_test_scaled)
y_pred2 = model2.predict(X_test_scaled)
y_pred_ensemble = (y_pred1 + y_pred2) / 2
```

**方法2：加权平均**
```python
# 根据验证集性能加权
weight1 = 0.6
weight2 = 0.4
y_pred_ensemble = weight1 * y_pred1 + weight2 * y_pred2
```

---

## 相关文档

- [数据处理工具库](dataframe-reference.md) - 特征计算所需的工具函数
- [因子分析工具](factor-analysis-reference.md) - 特征和标签生成方法
- [数据参考](data-reference.md) - 获取特征所需的数据
- [回测参考](backtesting-reference.md) - ML 策略回测
- [最佳实践](best-practices.md) - 避免常见错误

---

## 总结

机器学习在量化投资中的应用需要严谨的工作流程和方法论。本文档提供了：

1. **完整的 ML 工作流程**：从数据准备到回测验证的10个步骤
2. **特征工程模板**：技术指标特征、基本面特征、标签生成方法
3. **3个完整示例**：scikit-learn 随机森林、XGBoost 梯度提升、特征选择和交叉验证
4. **最佳实践**：避免常见错误、特征工程技巧、模型调优技巧

**记住**：
- 不要穷举所有 ML 方法，而是学会使用 Context7 查询
- 不要过度拟合历史数据，而是注重模型的泛化能力
- 不要忽视交易成本和市场冲击，而是进行真实的回测验证
- 始终使用时间序列划分，避免数据泄露

**下一步**：
- 使用本文档的模板开发 ML 策略
- 参考 `factor-analysis-reference.md` 进行特征评估
- 参考 `backtesting-reference.md` 进行策略回测
- 使用 Context7 查询 scikit-learn、XGBoost 等库的详细文档
