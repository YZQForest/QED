
#  数学推理智能体

面向 **2026 年度中国青年科技创新“揭榜挂帅”擂台赛 · 书生赛道** 赛题“基于 Intern-S 系列大模型的数学智能体设计与推理创新”。

本仓库为最终评测版本，入口、依赖和返回结构均按赛事要求实现。

## 仓库结构

```text
.
├── user_agent.py      # 平台强制入口：ReasoningAgent
├── requirements.txt   # 项目依赖：sympy>=1.12
├── 提交说明.md         # 队伍、仓库、模型与提交信息
├── 项目介绍书.md       # 项目背景、方案与模块说明
├── 测试报告.md         # 本地自测与合规验证结果
└── 提交前自检清单.md   # 推送 main 分支前的最终核对项
```

## 平台调用方式

```python
from user_agent import ReasoningAgent

agent = ReasoningAgent(client=official_client)
result = agent.solve(problem="题目文本", metadata={"idx": 0})
```

返回结果：

```python
{
    "final_response": "最终答案字符串",
    "trace": [
        {"step": "solve", "content": "关键推理与校验信息"}
    ]
}
```

## 技术方案概要

1. **多候选生成**：多温度并发采样多条独立推理路径，降低单次采样误差。
2. **答案解析**：支持 `答案：X`、`\boxed{}`、`Final Answer`、选择题字母、末行表达式和截断回捞等多种格式。
3. **符号等价聚类**：使用 SymPy 对分数、小数、根式、复数、集合与多值答案进行等价判断。
4. **自洽投票**：按等价答案聚类并进行加权多数表决；SymPy 验算结果作为独立票源。
5. **条件化纠错**：在无多数、平票、解析失败或答案契约失败时触发反思重推导或 LLM 裁判。
6. **输出契约化**：按选择题、多选题、证明题、判断题、敛散性、假设检验、区间和一题多解题分别规范化最终答案。
7. **平台兼容**：优先尝试 `enable_thinking=False`，客户端不支持时自动降级；全程不硬编码密钥。

## 本地安装

```bash
pip install -r requirements.txt
```

## 合规说明

- `user_agent.py` 位于仓库根目录；
- 固定实现 `ReasoningAgent(client, *args, **kwargs)`；
- 固定实现 `solve(problem: str, metadata: dict) -> dict`；
- `final_response` 保证为非空可读字符串；
- `trace` 为数组，整体返回可 JSON 序列化；
- 不读取测试集标准答案，不包含硬编码 API Key，不依赖本机绝对路径。
