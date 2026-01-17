# AShare AI

A股量化交易 Claude Code Skill - 使用 Tushare Pro 获取数据，Backtrader 进行回测

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Skill-blueviolet)](https://claude.ai)

## 快速安装

### 对于 Claude Code 用户

直接告诉 Claude：

```
请安装 https://github.com/410417122/ashare-ai
```

就这么简单，Claude 会自动处理安装。

## 功能特性

- **Tushare Pro 数据** — 完整的 A股市场数据，包括行情、财务、资金流向等
- **Backtrader 回测** — 专业的回测引擎，支持止损止盈、仓位管理、风险控制
- **因子分析** — IC 分析、Shapley 值、中心性指标
- **策略示例** — 60+ 完整的量化策略示例
- **机器学习** — 特征工程和标签生成

## 使用示例

### 获取股票数据

**提示词：**
> "获取平安银行（000001.SZ）最近 30 天的日线数据"

### 回测策略

**提示词：**
> "回测一个简单的均线策略：5日均线上穿20日均线买入，下穿卖出"

### 因子分析

**提示词：**
> "分析 PE、PB、ROE 这三个因子对收益率的预测能力"

## 核心原则

### 执行哲学

当你要求回测时，AI 会**执行代码并展示结果**，而不是让你自己复制代码运行。

当你要求图表时，AI 会**生成并打开图片**，而不是只给你一个文件路径。

### 文档查询

遇到不确定的 Tushare 接口或 Backtrader 参数时，AI 会自动使用 Context7 查询官方文档，确保信息准确。

## 技术栈

- **数据源**: Tushare Pro
- **回测引擎**: Backtrader
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, mplfinance

## 文档结构

```
ashare-plugin/
├── .claude-plugin/
│   └── plugin.json          # 插件配置
└── skills/ashare/
    ├── SKILL.md             # 主入口文件
    ├── data-reference.md    # Tushare 数据接口参考
    ├── factor-examples.md   # 60+ 策略示例
    ├── dataframe-reference.md  # DataFrame 工具参考
    ├── factor-analysis-reference.md  # 因子分析参考
    ├── machine-learning-reference.md  # 机器学习参考
    ├── best-practices.md    # 最佳实践
    ├── debugging-guide.md   # 调试指南
    ├── known-issues.md      # 已知问题
    ├── context7-guide.md    # Context7 使用指南
    ├── data-bridge.md       # 数据桥接指南
    └── ashare-rules.md      # A股特殊规则
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 相关项目

本项目基于 [FinLab AI](https://github.com/koreal6803/finlab-ai) 改造，感谢原作者的贡献。
