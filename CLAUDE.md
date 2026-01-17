# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AShare AI 是一个 Claude Code Skill，提供 A股量化交易功能。使用 Tushare Pro 获取数据，Backtrader 进行回测。

## 架构

```
ashare-ai/
├── ashare-plugin/           # Claude Code 插件
│   ├── .claude-plugin/      # 插件元数据
│   │   └── plugin.json      # 插件配置
│   └── skills/ashare/       # Skill 文档 (*.md 文件)
│       ├── SKILL.md         # 主入口文件
│       ├── data-reference.md
│       ├── factor-examples.md
│       ├── dataframe-reference.md
│       ├── factor-analysis-reference.md
│       ├── machine-learning-reference.md
│       ├── best-practices.md
│       ├── debugging-guide.md
│       ├── known-issues.md
│       ├── context7-guide.md
│       ├── data-bridge.md
│       └── ashare-rules.md
├── .claude-plugin/          # 市场注册
│   └── marketplace.json
├── README.md
├── CLAUDE.md
└── .gitignore
```

## 安装方法

### 全局安装（推荐）

```bash
# 用户只需告诉 Claude：
# "请全局安装 https://github.com/410417122/ashare-ai"

# Claude 会自动执行：
# 1. 添加 marketplace
claude plugin marketplace add 410417122/ashare-ai

# 2. 安装插件（默认为全局安装）
claude plugin install ashare-plugin@ashare-plugins
```

全局安装后，在任何项目中都可以使用 `/ashare` skill，无需重复安装。

### 更新插件

```bash
# 更新 marketplace
claude plugin marketplace update ashare-plugins

# 更新插件
claude plugin update ashare-plugin@ashare-plugins
```

## 文档文件

所有 skill 文档位于 `ashare-plugin/skills/ashare/`：

| 文件 | 用途 |
|------|------|
| SKILL.md | 主入口、执行哲学、快速开始 |
| data-reference.md | Tushare Pro 数据接口参考 |
| factor-examples.md | 60+ 完整策略示例 |
| dataframe-reference.md | DataFrame 工具方法 |
| factor-analysis-reference.md | 因子分析方法 |
| machine-learning-reference.md | 机器学习功能 |
| best-practices.md | 最佳实践、反模式预防 |
| debugging-guide.md | 调试指南和常见问题 |
| known-issues.md | 已知问题和解决方案 |
| context7-guide.md | Context7 文档查询指南 |
| data-bridge.md | 数据桥接和转换 |
| ashare-rules.md | A股特殊规则（T+1、涨跌停等）|

## 执行哲学

来自 SKILL.md - 当用户请求回测或数据时：
- 执行代码并展示结果，不要让用户自己运行代码
- 生成文件后自动打开
- 遇到错误要调试修复，不要只报告错误
- 不确定的接口必须用 Context7 查询官方文档

## 核心原则

### 1. 不确定就查文档

遇到任何不确定的 Tushare 接口、参数、字段，必须用 Context7 查询官方文档：

```
Library ID: /websites/tushare_pro_document

查询示例：
- "daily 日线行情 返回字段 open high low close"
- "fina_indicator 财务指标 roe eps 参数"
- "stock_basic 股票列表 industry market"
```

### 2. 执行而非教学

用户要求回测时，要执行代码展示结果，而不是给代码让用户自己运行。

### 3. A股特殊规则

- T+1 交易制度
- 涨跌停限制（10% 或 20%）
- ST 股票特殊处理
- 交易时间和节假日

## 技术栈

- **数据源**: Tushare Pro
- **回测引擎**: Backtrader
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, mplfinance

## 语言

根据用户语言响应。中文提问用中文回答。
