"""
HANS-MAS v4.0: Hierarchical Adaptive Neuro-Symbolic Multi-Agent System
Challenge-Cup-2026 书生赛道 - 数学推理智能体

版本: 4.0 (2026-07-24 全面优化版)
核心优化:
- s2-preview Thinking Token截断修复 (英文prompt约束 + 答案回捞)
- 答案契约检查 (AnswerContract - 借鉴ICMA)
- Token预算控制 (TokenBudget - 借鉴ICMA)
- ReAct风格Python沙箱执行 (ReActPythonExecutor - 借鉴MathRouterAgent)
- 增强AnswerParser (支持Thinking Process输出格式)
- 答案格式化回捞 (AnswerRecovery)
- 上下文裁剪 (ContextPruner)
- 幻觉检测 (HallucinationDetector)
- 验证器打分 (LLMVerifier - 借鉴MathRouterAgent)
- 代码截断恢复 (CodeTruncationRecovery)

官方接口兼容:
- 类名: ReasoningAgent
- 构造: ReasoningAgent(client)
- 方法: solve(problem: str, metadata: dict) -> dict
- 返回: {"final_response": str, "trace": list}

依赖:
- Python >= 3.7
- sympy >= 1.12
"""

import re
import time
import signal
import threading
import platform
import concurrent.futures
import ast
import textwrap
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from collections import Counter

# =============================================================================
# 配置类 - 用户可控参数 (v4.0新增模型适配配置)
# =============================================================================

@dataclass
class AgentConfig:
    """推理Agent配置 - 针对s1/s2模型差异自动适配"""

    # 思考模式: quick(快速) / standard(标准) / deep(深度)
    thinking_mode: str = "standard"

    # 验证严格程度: loose / medium / strict
    validation_strictness: str = "medium"

    # SymPy计算超时(秒)
    sympy_timeout: float = 8.0

    # LLM最大token数 (v4.0: s2模型预留40%给答案输出)
    max_tokens: int = 4096

    # 生成候选时的温度策略
    policy_temperature: float = 0.4

    # 单题超时(秒)
    problem_timeout: int = 90

    # 是否启用离线RAG知识检索
    enable_rag: bool = True

    # 是否启用thinking_mode API字段
    enable_thinking: bool = False  # v4.0: s2模型thinking消耗token过多，默认关闭

    # RPM速率限制
    rpm_limit: int = 25

    # LLM调用重试次数
    max_llm_retries: int = 3

    # 重试基础延迟(秒)
    retry_base_delay: float = 1.0

    # v4.0 NEW: Token预算控制 (借鉴ICMA)
    enable_token_budget: bool = True
    token_budget_ratio: float = 0.85  # 使用预算的85%后进入保守模式

    # v4.0 NEW: 答案契约检查
    enable_answer_contract: bool = True

    # v4.0 NEW: ReAct Python执行
    enable_react_python: bool = True
    react_max_steps: int = 10
    react_timeout_per_step: float = 15.0

    # v4.0 NEW: 验证器打分
    enable_verifier_scoring: bool = True

    # v4.0 NEW: 答案回捞
    enable_answer_recovery: bool = True

    # v4.0 NEW: 上下文裁剪
    enable_context_pruning: bool = True
    max_context_messages: int = 20

    # v4.0 NEW: 幻觉检测
    enable_hallucination_detection: bool = True

    # v4.0 NEW: 代码截断恢复
    enable_truncation_recovery: bool = True

    # v4.0 NEW: 模型类型检测 (auto/s1/s2)
    model_type: str = "auto"

    def adapt_for_model(self, model_name: str):
        """根据模型名称自动调整配置"""
        if "s2" in model_name.lower():
            self.model_type = "s2"
            self.enable_thinking = False  # s2的thinking消耗太多token
            self.max_tokens = 4096
            # s2需要更严格的prompt约束来减少Thinking Process
        elif "s1" in model_name.lower():
            self.model_type = "s1"
            self.enable_thinking = True
            self.max_tokens = 4096
        else:
            self.model_type = "auto"


# =============================================================================
# 18个数学子领域知识库 (v4.0: 扩充并结构化)
# =============================================================================

MATH_KNOWLEDGE_BASE = {
    "discrete_mathematics": {
        "name": "离散数学",
        "weight": 0.2143,
        "keywords": ["鸽巢原理", "容斥原理", "排列组合", "二项式定理", "Catalan数",
                     "Burnside引理", "Polya计数", "图论", "集合", "递推", "生成函数",
                     "组合", "排列", "计数", "图", "顶点", "边", "路径", "连通",
                     "鸽巢", "抽屉", "容斥", "Stirling数", "Bell数", "母函数"],
        "theorems": [
            "鸽巢原理: n+1个物品放入n个盒子，至少一个盒子有>=2个物品",
            "容斥原理: |A∪B| = |A|+|B|-|A∩B|，推广到n个集合",
            "二项式定理: (a+b)^n = ΣC(n,k)a^(n-k)b^k",
            "Catalan数: C_n = (2n)!/((n+1)!n!)，括号化、路径计数",
            "Burnside引理: 轨道数 = (1/|G|)Σ|Fix(g)|",
            "Polya计数定理: 带权Burnside引理，染色计数",
        ],
        "formulas": [
            "排列数: P(n,k) = n!/(n-k)!",
            "组合数: C(n,k) = n!/(k!(n-k)!)",
            "Stirling数: S(n,k) = (1/k!)Σ(-1)^(k-j)C(k,j)j^n",
            "Bell数: B_n = ΣS(n,k)",
            "生成函数: A(x) = Σa_n x^n",
        ],
        "strategies": [
            "计数问题: 先分类再分步，注意去重和对称性",
            "图论问题: 考虑连通性、度序列、欧拉/哈密顿路径",
            "逻辑问题: 使用真值表或逻辑等价变换",
            "集合问题: 善用容斥原理和文氏图",
            "递推问题: 建立递推关系，用生成函数求解",
        ],
        "sympy_ops": ["binomial", "factorial", "permutations", "partitions"],
    },
    "numerical_analysis": {
        "name": "数值分析",
        "weight": 0.1161,
        "keywords": ["数值积分", "梯形法则", "Simpson法则", "插值", "Lagrange",
                     "Newton迭代", "误差分析", "Richardson外推", "Runge现象",
                     "数值微分", "二分法", "Newton法", "不动点迭代", "收敛阶"],
        "theorems": [
            "Taylor展开: f(x) = Σf^(n)(a)/n!·(x-a)^n + R_n(x)",
            "Lagrange插值: 唯一n次多项式过n+1个不同点",
            "Newton迭代: x_{n+1} = x_n - f(x_n)/f'(x_n), 局部二次收敛",
        ],
        "formulas": [
            "梯形法则: ∫f ≈ (b-a)/2·[f(a)+f(b)]",
            "Simpson法则: ∫f ≈ (b-a)/6·[f(a)+4f((a+b)/2)+f(b)]",
            "复合梯形误差: |E| <= M_2/(12n^2)·(b-a)^3",
            "Richardson外推: 不同步长线性组合提高精度",
        ],
        "strategies": [
            "数值积分: 先分析被积函数光滑性",
            "插值问题: 注意Runge现象，高次插值不稳定",
            "方程求解: 先用图形法/二分法估计根的位置",
            "误差分析: 区分截断误差和舍入误差",
        ],
        "sympy_ops": ["nsimplify", "N", "integrate"],
    },
    "measure_integration": {
        "name": "测度积分",
        "weight": 0.0982,
        "keywords": ["Lebesgue积分", "控制收敛定理", "Fubini定理", "单调收敛",
                     "Fatou引理", "可测函数", "可积", "几乎处处", "L^p空间"],
        "theorems": [
            "Lebesgue控制收敛: |f_n|<=g且g可积，则lim∫f_n = ∫lim f_n",
            "Fubini定理: 重积分化为累次积分(要求绝对可积)",
            "单调收敛定理: 0<=f_n↗f，则∫f_n → ∫f",
            "Fatou引理: ∫liminf f_n <= liminf ∫f_n",
        ],
        "strategies": ["判断可积性: 先检查可测性，再找控制函数",
                       "计算Lebesgue积分: 连续函数可转化为Riemann积分"],
        "sympy_ops": ["integrate", "limit"],
    },
    "differential_geometry": {
        "name": "微分几何",
        "weight": 0.0804,
        "keywords": ["Gauss-Bonnet", "Stokes定理", "曲率", "挠率", "第一基本形式",
                     "第二基本形式", "测地线", "流形", "切空间"],
        "theorems": [
            "Gauss-Bonnet: 闭曲面的总曲率 = 2πχ(M)",
            "Stokes定理: ∫_M dω = ∫_{∂M} ω",
        ],
        "sympy_ops": ["diff", "simplify", "Matrix"],
    },
    "probability_theory": {
        "name": "概率论",
        "weight": 0.0714,
        "keywords": ["全概率公式", "Bayes公式", "大数定律", "中心极限定理",
                     "期望", "方差", "协方差", "特征函数", "分布函数"],
        "theorems": [
            "全概率公式: P(A) = ΣP(A|B_i)P(B_i)",
            "Bayes公式: P(B_i|A) = P(A|B_i)P(B_i)/P(A)",
            "大数定律: 样本均值收敛于期望",
            "中心极限定理: (S_n-nμ)/(σ√n) → N(0,1)",
        ],
        "sympy_ops": ["integrate", "summation"],
    },
    "abstract_algebra": {
        "name": "抽象代数",
        "weight": 0.0714,
        "keywords": ["Lagrange定理", "Cayley定理", "Sylow定理", "有限域",
                     "群", "环", "域", "同态", "同构", "正规子群", "商群"],
        "theorems": [
            "Lagrange定理: 子群H的阶整除群G的阶",
            "Cayley定理: 任何群同构于某置换群的子群",
            "Sylow定理: p^k ||G| 则存在Sylow p-子群",
            "有限域: F_{p^n}存在且唯一，乘法群循环",
        ],
        "sympy_ops": ["gf_factor", "groebner"],
    },
    "stochastic_process": {
        "name": "随机过程",
        "weight": 0.0625,
        "keywords": ["Markov链", "Poisson过程", "Brown运动", "鞅",
                     "转移概率", "平稳分布", "常返", "瞬过"],
        "sympy_ops": ["summation", "limit"],
    },
    "complex_analysis": {
        "name": "复分析",
        "weight": 0.0625,
        "keywords": ["Cauchy积分公式", "留数定理", "解析函数", "全纯",
                     "Laurent级数", "共形映射", "辐角原理", "Rouché定理"],
        "theorems": [
            "Cauchy积分公式: f(z) = (1/2πi)∮f(ζ)/(ζ-z)dζ",
            "留数定理: ∮f(z)dz = 2πi·ΣRes(f,a_k)",
        ],
        "sympy_ops": ["residue", "integrate"],
    },
    "ode": {
        "name": "常微分方程",
        "weight": 0.0536,
        "keywords": ["ODE", "常微分方程", "分离变量", "积分因子", "特征方程",
                     "Wronskian", "Laplace变换", "级数解", "边值问题"],
        "sympy_ops": ["dsolve", "laplace_transform"],
    },
    "statistical_inference": {
        "name": "统计推断",
        "weight": 0.0536,
        "keywords": ["极大似然估计", "矩估计", "假设检验", "置信区间",
                     "t检验", "卡方检验", "F检验", "p值", "拒绝域"],
        "sympy_ops": ["integrate", "diff"],
    },
    "functional_analysis": {
        "name": "泛函分析",
        "weight": 0.0446,
        "keywords": ["Banach空间", "Hilbert空间", "算子", "泛函",
                     "对偶空间", "谱理论", "紧算子", "弱收敛"],
        "sympy_ops": ["integrate", "limit"],
    },
    "linear_regression": {
        "name": "线性回归",
        "weight": 0.0446,
        "keywords": ["最小二乘法", "OLS", "R²", "残差", "多重共线性",
                     "岭回归", "Lasso", "回归系数", "显著性检验"],
        "sympy_ops": ["Matrix", "solve"],
    },
    "pde": {
        "name": "偏微分方程",
        "weight": 0.0357,
        "keywords": ["PDE", "分离变量法", "特征线法", "Green函数",
                     "热方程", "波动方程", "Laplace方程", "边值问题"],
        "sympy_ops": ["pdsolve", "dsolve"],
    },
    "advanced_algebra": {
        "name": "高等代数",
        "weight": 0.0357,
        "keywords": ["Jordan标准型", "Smith标准型", "特征多项式",
                     "最小多项式", "不变子空间", "根子空间", "若尔当块"],
        "sympy_ops": ["jordan_cells", "charpoly"],
    },
    "operations_research": {
        "name": "运筹学",
        "weight": 0.0268,
        "keywords": ["线性规划", "单纯形法", "对偶问题", "整数规划",
                     "动态规划", "最短路", "网络流", "运输问题"],
        "sympy_ops": ["linear_eq_to_matrix", "linsolve"],
    },
    "mathematical_analysis": {
        "name": "数学分析",
        "weight": 0.0268,
        "keywords": ["一致收敛", "等度连续", "Arzelà-Ascoli", "紧性",
                     "完备性", "连续", "一致连续", "可微", "可积"],
        "sympy_ops": ["limit", "diff", "integrate"],
    },
    "topology": {
        "name": "拓扑学",
        "weight": 0.0179,
        "keywords": ["连通性", "紧致性", "Hausdorff", "同胚",
                     "基本群", "覆叠空间", "同调群", "流形"],
        "sympy_ops": [],
    },
    "general": {
        "name": "通用数学",
        "weight": 0.01,
        "keywords": ["计算", "求解", "证明", "推导", "化简"],
        "sympy_ops": ["simplify", "expand", "factor"],
    },
}


# =============================================================================
# v4.0 NEW: 答案契约检查 (借鉴ICMA)
# =============================================================================

class AnswerContract:
    """从题面推断答案必须包含的组件，用于检测答案完整性"""

    # 题型→必须包含的正则模式
    CONTRACT_PATTERNS = {
        "hypothesis_test": {
            "keywords": ["假设检验", "检验", "显著性水平", "拒绝", "不拒绝",
                         "H0", "H_0", "零假设", "原假设", "p值"],
            "required": [r"拒绝|不拒绝|接受|无法拒绝|fail to reject|reject"],
            "description": "假设检验题必须包含是否拒绝零假设的结论"
        },
        "extrema": {
            "keywords": ["极值", "极大", "极小", "最大值", "最小值", "驻点", "临界点"],
            "required": [r"极大|极小|最大|最小|maximum|minimum"],
            "description": "求极值题必须包含极大/极小的判断"
        },
        "convergence": {
            "keywords": ["收敛", "发散", "级数", "判别法", "敛散性"],
            "required": [r"收敛|发散|converge|diverge"],
            "description": "级数敛散性题必须包含收敛或发散的结论"
        },
        "confidence_interval": {
            "keywords": ["置信区间", "置信度", "置信水平", "区间估计"],
            "required": [r"置信区间|\\[.*,.*\\]|\\(.*,.*\\)|confidence interval"],
            "description": "置信区间题必须包含区间表示"
        },
        "probability": {
            "keywords": ["概率", "求P\\(", "求概率", "条件概率"],
            "required": [r"P\\s*\\(|概率|probability"],
            "description": "概率题应包含概率值或表达式"
        },
        "counting": {
            "keywords": ["多少种", "个数", "计数", "排列数", "组合数"],
            "required": [r"\\d+|\\\\binom|C\\(|P\\("],
            "description": "计数题应包含具体的数字答案"
        },
        "field_order": {
            "keywords": ["有限域", "\\\\mathbb{F}", "\\mathbb{F}", "域的阶", "元素个数"],
            "required": [r"\\d+|\\\\mathbb{F}"],
            "description": "有限域题应包含阶数或元素个数"
        },
    }

    @classmethod
    def analyze(cls, problem: str, answer: str) -> Dict[str, Any]:
        """分析答案是否满足题面契约
        Returns: {"passed": bool, "missing": List[str], "details": List[Dict]}
        """
        problem_lower = problem.lower()
        missing = []
        details = []

        for contract_type, info in cls.CONTRACT_PATTERNS.items():
            # 检查题面是否匹配该契约类型
            matched = any(kw in problem_lower or kw in problem for kw in info["keywords"])
            if not matched:
                continue

            # 检查答案是否包含必需的组件
            found = False
            for pattern in info["required"]:
                if re.search(pattern, answer, re.IGNORECASE):
                    found = True
                    break

            status = "PASS" if found else "MISSING"
            details.append({
                "type": contract_type,
                "description": info["description"],
                "status": status
            })
            if not found:
                missing.append(info["description"])

        return {
            "passed": len(missing) == 0,
            "missing": missing,
            "details": details,
            "contract_count": len(details)
        }


# =============================================================================
# v4.0 NEW: Token预算控制 (借鉴ICMA)
# =============================================================================

class TokenBudget:
    """粗粒度Token预算控制 - 避免超出平台限制"""

    # 预估各操作消耗的token数 (经验值)
    COST_ESTIMATES = {
        "analysis_call": 800,      # 问题分析调用
        "candidate_generation": 3500,  # 单次候选生成 (s2模型thinking消耗大)
        "verifier_scoring": 600,   # 验证器打分
        "sympy_execution": 50,     # SymPy执行 (本地，几乎不消耗)
        "answer_formatting": 200,  # 答案格式化
    }

    def __init__(self, total_budget: int = 10000):
        self.total_budget = total_budget
        self.used_tokens = 0
        self.call_log: List[Dict] = []
        self.tight_mode = False  # 保守模式

    def estimate_call(self, call_type: str, num_calls: int = 1) -> int:
        """预估一次调用的token消耗"""
        base = self.COST_ESTIMATES.get(call_type, 1000)
        return base * num_calls

    def consume(self, call_type: str, actual_tokens: Optional[int] = None,
                num_calls: int = 1):
        """记录token消耗"""
        estimated = self.estimate_call(call_type, num_calls)
        consumed = actual_tokens if actual_tokens else estimated
        self.used_tokens += consumed
        self.call_log.append({
            "type": call_type,
            "estimated": estimated,
            "actual": consumed,
            "cumulative": self.used_tokens
        })

        # 检查是否进入保守模式
        ratio = self.used_tokens / self.total_budget
        if ratio > 0.85 and not self.tight_mode:
            self.tight_mode = True

    def can_afford(self, call_type: str, num_calls: int = 1) -> bool:
        """检查是否还能负担指定调用"""
        estimated = self.estimate_call(call_type, num_calls)
        return (self.used_tokens + estimated) <= self.total_budget

    def remaining(self) -> int:
        return max(0, self.total_budget - self.used_tokens)

    def get_summary(self) -> Dict:
        return {
            "total_budget": self.total_budget,
            "used": self.used_tokens,
            "remaining": self.remaining(),
            "usage_ratio": self.used_tokens / self.total_budget,
            "tight_mode": self.tight_mode,
            "call_count": len(self.call_log)
        }


# =============================================================================
# v4.0 NEW: 上下文裁剪 (借鉴MathRouterAgent)
# =============================================================================

class ContextPruner:
    """防止消息历史过长导致INPUT_LENGTH_ERROR"""

    MAX_MESSAGES = 22  # 保留system + 首条user + 最近N轮
    KEEP_RECENT = 5    # 保留最近N轮对话

    @classmethod
    def prune(cls, messages: List[Dict]) -> List[Dict]:
        """裁剪消息列表，保留关键上下文"""
        if len(messages) <= cls.MAX_MESSAGES:
            return messages

        # 保留第一条(system)和第二条(首条user)
        pruned = messages[:2] if len(messages) >= 2 else messages[:1]

        # 保留最近的KEEP_RECENT轮
        recent_start = max(2, len(messages) - cls.KEEP_RECENT * 2)
        pruned.extend(messages[recent_start:])

        return pruned


# =============================================================================
# v4.0 NEW: 幻觉检测 (借鉴MathRouterAgent)
# =============================================================================

class HallucinationDetector:
    """检测模型自行模拟的执行结果"""

    # 模型可能自行生成的假执行标记
    FAKE_EXECUTION_PATTERNS = [
        r'```\s*\n*\s*output\s*[:：]',
        r'\[\s*执行结果\s*[:：]',
        r'\[\s*计算结果\s*[:：]',
        r'模拟执行',
        r'假想的执行结果',
    ]

    @classmethod
    def detect(cls, text: str) -> Dict:
        """检测文本中是否包含幻觉执行结果"""
        findings = []
        for pattern in cls.FAKE_EXECUTION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                findings.append({
                    "pattern": pattern,
                    "match": match.group(0),
                    "position": match.start()
                })

        return {
            "has_hallucination": len(findings) > 0,
            "findings": findings,
            "count": len(findings)
        }


# =============================================================================
# v4.0 NEW: 代码截断恢复 (借鉴MathRouterAgent)
# =============================================================================

class CodeTruncationRecovery:
    """通过AST解析恢复被max_tokens截断的Python代码"""

    @staticmethod
    def recover(code: str) -> str:
        """尝试恢复截断的Python代码"""
        # 策略1: 逐行尝试，找到最长可解析前缀
        lines = code.split('\n')
        best_recovery = ""

        for i in range(len(lines), 0, -1):
            candidate = '\n'.join(lines[:i])
            if CodeTruncationRecovery._is_valid_python(candidate):
                best_recovery = candidate
                break

        # 策略2: 如果是单行表达式，尝试补全括号
        if not best_recovery and code.strip():
            balanced = CodeTruncationRecovery._balance_brackets(code.strip())
            if CodeTruncationRecovery._is_valid_python(balanced):
                best_recovery = balanced

        # 策略3: 如果代码以运算符结尾，截断到上一个完整表达式
        if not best_recovery:
            best_recovery = CodeTruncationRecovery._truncate_to_last_expr(code)

        return best_recovery if best_recovery else code

    @staticmethod
    def _is_valid_python(code: str) -> bool:
        """检查代码是否可被AST解析"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _balance_brackets(expr: str) -> str:
        """平衡括号"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        for c in expr:
            if c in pairs:
                stack.append(c)
            elif c in pairs.values():
                if stack and pairs[stack[-1]] == c:
                    stack.pop()
        # 补全未闭合的括号
        for opener in reversed(stack):
            expr += pairs[opener]
        return expr

    @staticmethod
    def _truncate_to_last_expr(code: str) -> str:
        """截断到最后一个完整表达式"""
        # 找到最后一个完整的赋值或表达式
        lines = code.split('\n')
        for i in range(len(lines) - 1, -1, -1):
            candidate = '\n'.join(lines[:i+1]).rstrip()
            # 移除末尾的未闭合行继续符
            candidate = re.sub(r'[\\+\-*/=,\(]+$[\s]*$', '', candidate)
            if CodeTruncationRecovery._is_valid_python(candidate):
                return candidate
        return code


# =============================================================================
# 跨平台超时保护
# =============================================================================

class TimeoutContext:
    """跨平台超时保护 (Linux: SIGALRM / Windows: threading)"""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._timeout_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._original_handler = None
        self._is_windows = platform.system() == 'Windows'

    def __enter__(self):
        self._timeout_event.clear()
        if self._is_windows:
            self._thread = threading.Thread(target=self._windows_timer)
            self._thread.daemon = True
            self._thread.start()
        else:
            self._original_handler = signal.signal(
                signal.SIGALRM, self._unix_handler
            )
            signal.alarm(int(self.seconds) + 1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_windows:
            self._timeout_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1)
        else:
            signal.alarm(0)
            if self._original_handler:
                signal.signal(signal.SIGALRM, self._original_handler)

    def _windows_timer(self):
        if not self._timeout_event.wait(timeout=self.seconds):
            raise TimeoutError(f"Operation timed out after {self.seconds}s")

    def _unix_handler(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.seconds}s")

    def check_timeout(self) -> bool:
        """检查是否已超时 (Windows下使用)"""
        return self._timeout_event.is_set()


# =============================================================================
# SymPy安全沙箱 (v4.0: 增加截断恢复支持)
# =============================================================================

class SafeSymPyExecutor:
    """安全的SymPy表达式执行器 (静态检查+受限环境+超时)"""

    ALLOWED_NAMES = {
        'abs', 'all', 'any', 'bin', 'bool', 'chr', 'complex', 'dict', 'divmod',
        'enumerate', 'filter', 'float', 'format', 'frozenset', 'hasattr',
        'hex', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list',
        'map', 'max', 'min', 'next', 'oct', 'ord', 'pow', 'range', 'repr',
        'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum', 'tuple',
        'type', 'vars', 'zip', 'True', 'False', 'None',
        # SymPy核心
        'sympy', 'sp', 'Symbol', 'symbols', 'simplify', 'expand', 'factor',
        'solve', 'diff', 'integrate', 'limit', 'summation', 'product',
        'oo', 'pi', 'E', 'I', 'exp', 'log', 'sin', 'cos', 'tan',
        'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'sqrt',
        'Rational', 'Integer', 'Float', 'Matrix', 'Determinant',
        'binomial', 'factorial', 'gamma', 'zeta', 'fibonacci',
        'nsimplify', 'N', 'evalf', 'expand_trig', 'expand_log',
        'apart', 'together', 'cancel', 'degree', 'roots', 'groebner',
        # 数值计算
        'float', 'int', 'complex',
    }

    BLOCKED_PATTERNS = [
        r'\b__\w+__\b', r'\bimport\b', r'\bopen\b', r'\beval\b',
        r'\bexec\b', r'\bcompile\b', r'\bsubprocess\b', r'\bsocket\b',
        r'\burllib\b', r'\brequests\b', r'\bhttp\b', r'\bftp\b',
        r'\bos\.', r'\bsys\.', r'\bshutil\.', r'\bpathlib\b',
        r'[\'"]\.\s*[\'"]',  # 目录遍历
    ]

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self.stats = {"total": 0, "success": 0, "timeout": 0, "error": 0}

    def _is_safe(self, code: str) -> bool:
        """静态代码安全检查"""
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in self.ALLOWED_NAMES:
                return False
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id not in self.ALLOWED_NAMES:
                    return False
        return True

    def execute(self, code: str) -> Dict[str, Any]:
        """执行SymPy表达式并返回结果"""
        self.stats["total"] += 1
        code = code.strip()
        if not code:
            return {"success": False, "error": "Empty code", "result": None}

        # v4.0: 代码截断恢复
        if not self._is_safe(code):
            recovered = CodeTruncationRecovery.recover(code)
            if recovered and self._is_safe(recovered):
                code = recovered
            else:
                self.stats["error"] += 1
                return {"success": False, "error": "Unsafe code blocked", "result": None}

        safe_globals = {name: __import__('sympy').__dict__.get(name, __builtins__.get(name))
                        for name in self.ALLOWED_NAMES if name in __import__('sympy').__dict__ or name in __builtins__}
        safe_globals.update({
            'sympy': __import__('sympy'),
            'sp': __import__('sympy'),
            '__builtins__': {},
        })

        try:
            with TimeoutContext(self.timeout):
                result = eval(compile(code, '<string>', 'eval'), safe_globals, {})
                self.stats["success"] += 1
                return {"success": True, "result": result, "error": None}
        except TimeoutError:
            self.stats["timeout"] += 1
            return {"success": False, "error": "Timeout", "result": None}
        except Exception as e:
            self.stats["error"] += 1
            return {"success": False, "error": str(e), "result": None}


# =============================================================================
# LLMClient适配层
# =============================================================================

class LLMClient:
    """兼容InternChatClient和OpenAI接口的适配层"""

    def __init__(self, client):
        self._client = client
        self._model_name = getattr(client, 'model', None)

    def chat(self, messages, temperature=0.5, max_tokens=4096, **kwargs):
        """统一调用接口"""
        if hasattr(self._client, 'chat'):
            return self._client.chat(messages, temperature, max_tokens, **kwargs)
        elif hasattr(self._client, 'completions'):
            resp = self._client.completions.create(
                model=getattr(self._client, 'model', 'intern-s2-preview'),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return resp.choices[0].message.content
        elif callable(self._client):
            return self._client(messages, temperature, max_tokens)
        else:
            raise RuntimeError("Unsupported client type")

    def detect_model(self) -> str:
        """检测使用的模型名称"""
        if self._model_name:
            return self._model_name
        if hasattr(self._client, 'model'):
            return self._client.model
        return "intern-s2-preview"  # 默认


# =============================================================================
# 问题分析器
# =============================================================================

class ProblemAnalyzer:
    """分析问题类型、难度和所属领域"""

    QUESTION_TYPES = {
        "choice": {
            "keywords": ["选择", "下列", "选项", "A.", "B.", "C.", "D.",
                        "A、", "B、", "C、", "D、", "(A)", "(B)", "(C)", "(D)"],
            "weight": 1.0
        },
        "fill_blank": {
            "keywords": ["填空", "___", "______", "（ ）", "( )", "答案是",
                          "则", "那么", "因此", "可得", "由此可知"],
            "weight": 0.8
        },
        "proof": {
            "keywords": ["证明", "求证", "证:", "证明：", "请证明", "试证",
                          "证毕", "证法", "证明题", "demonstrate"],
            "weight": 1.0
        },
        "explanation": {
            "keywords": ["解释", "说明", "为什么", "原因", "理由",
                          "举例说明", "举例解释", "阐述"],
            "weight": 0.7
        },
        "computation": {
            "keywords": ["计算", "求", "求解", "算出", "结果", "值",
                          "等于", "evaluate", "compute", "calculate"],
            "weight": 0.9
        },
        "derivation": {
            "keywords": ["推导", "推出", "公式", "表达式", "化简", "变形"],
            "weight": 0.7
        },
    }

    def __init__(self):
        self.knowledge_base = MATH_KNOWLEDGE_BASE

    def analyze(self, problem: str) -> Dict[str, Any]:
        """分析问题并返回结构化信息"""
        subdomain_scores = self._detect_subdomain(problem)
        best_subdomain = max(subdomain_scores, key=subdomain_scores.get)

        qt_scores = self._detect_question_type(problem)
        best_qt = max(qt_scores, key=qt_scores.get)

        difficulty = self._assess_difficulty(problem)

        # 构建RAG上下文
        rag_context = self._build_rag_context(best_subdomain, problem)

        return {
            "subdomain": best_subdomain,
            "subdomain_scores": subdomain_scores,
            "question_type": best_qt,
            "question_type_scores": qt_scores,
            "difficulty": difficulty,
            "key_concepts": self._extract_concepts(problem),
            "rag_context": rag_context,
        }

    def _detect_subdomain(self, problem: str) -> Dict[str, float]:
        """加权关键词匹配检测数学子领域"""
        problem_lower = problem.lower()
        scores = {}
        for key, info in self.knowledge_base.items():
            score = sum(2 if kw in problem else (1 if kw in problem_lower else 0)
                       for kw in info.get("keywords", []))
            weighted = score * info.get("weight", 0.01)
            scores[key] = round(weighted, 4)
        if not any(scores.values()):
            scores["general"] = 0.1
        return scores

    def _detect_question_type(self, problem: str) -> Dict[str, float]:
        """检测题目类型"""
        problem_lower = problem.lower()
        scores = {}
        for qtype, info in self.QUESTION_TYPES.items():
            score = sum(1 for kw in info["keywords"] if kw in problem or kw in problem_lower)
            scores[qtype] = score * info["weight"]
        if not any(scores.values()):
            scores["computation"] = 0.1
        return scores

    def _assess_difficulty(self, problem: str) -> str:
        """评估题目难度"""
        difficulty_indicators = {
            "hard": ["证明", "任意", "所有", "一般", "任意给定", "抽象", "推广",
                     "一般地", "任意一个", "证明题", "综合题"],
            "easy": ["计算", "求值", "直接", "简单", "下列", "基础"],
        }
        scores = {"hard": 0, "medium": 0.5, "easy": 0}
        for level, words in difficulty_indicators.items():
            scores[level] += sum(1 for w in words if w in problem)
        return max(scores, key=scores.get)

    def _build_rag_context(self, subdomain: str, problem: str) -> str:
        """构建检索增强生成上下文"""
        info = self.knowledge_base.get(subdomain)
        if not info:
            return ""
        ctx_parts = [f"【{info['name']}】"]
        ctx_parts.extend(info.get("theorems", [])[:3])
        ctx_parts.extend(info.get("formulas", [])[:3])
        ctx_parts.extend(info.get("strategies", [])[:2])
        return "\n".join(ctx_parts)

    def _extract_concepts(self, problem: str) -> List[str]:
        """提取关键数学概念"""
        math_terms = re.findall(
            r'[\\]?[a-zA-Z]+\([^)]*\)|[\\]?[a-zA-Z]+_[a-zA-Z0-9]+|'
            r'[\u4e00-\u9fff]{2,8}(?:定理|公式|引理|性质|法则|原理|定义)',
            problem
        )
        return list(set(t.strip() for t in math_terms if len(t) > 1))[:8]


# =============================================================================
# v4.0: PromptEngineer - 针对s2模型优化，减少Thinking Token生成
# =============================================================================

class PromptEngineer:
    """构建高质量推理Prompt (v4.0: 针对s2-preview模型优化)"""

    # v4.0: 增加英文约束来减少s2模型的Thinking Process
    ANTI_THINKING_INSTRUCTIONS = """
【重要格式要求 - 必须严格遵守】
1. 直接输出计算过程和答案，**不要**输出Thinking Process或分析思路
2. 不要写"Thinking Process:"、"分析:"、"Step 1:"等前缀
3. 答案必须放在最后一行，格式为：答案：XXX
4. 如果题目是选择题，答案必须是A/B/C/D中的一个字母
5. 计算步骤要简洁，不要展开不必要的中间推理
""".strip()

    def build_prompt(self, problem: str, analysis: Dict) -> str:
        """构建推理Prompt"""
        parts = []
        parts.append("你是一位严谨的数学专家。请解决以下数学问题。")
        parts.append("")
        parts.append(self.ANTI_THINKING_INSTRUCTIONS)
        parts.append("")
        parts.append("【解题要求】")
        parts.append("1. 直接给出计算/推导/证明步骤")
        parts.append("2. 对于需要精确计算的地方，使用 [COMPUTE: 具体计算表达式] 标记")
        parts.append("3. 在最后明确写出最终答案，格式为：答案：XXX")
        parts.append('4. 答案不要使用\\boxed{}，直接写「答案：」后跟结果')
        parts.append("")

        rag = analysis.get("rag_context", "")
        if rag:
            parts.append("【相关背景知识】")
            parts.append(rag)
            parts.append("")

        qt = analysis.get("question_type", "computation")
        type_hints = {
            "choice": "这是一道选择题。请给出推理过程，最后答案必须是A/B/C/D中的一个字母。",
            "proof": '这是一道证明题。请给出完整的证明过程，最后说明「证毕」。',
            "computation": "这是一道计算题。请给出详细的计算过程和最终结果。",
            "fill_blank": "这是一道填空题。请给出计算过程，最后写出答案。",
        }
        parts.append(type_hints.get(qt, "请给出详细的解题过程和最终结果。"))
        parts.append("")
        parts.append("输出格式：")
        parts.append("计算过程：...")
        parts.append("答案：[最终结果]")
        parts.append("")
        parts.append(f"【题目】\n{problem}")

        return "\n".join(parts)

    def build_react_prompt(self, problem: str, analysis: Dict,
                           history: List[Dict] = None) -> str:
        """v4.0 NEW: 构建ReAct风格的推理Prompt"""
        parts = []
        parts.append("你是一位数学专家。请通过推理和Python代码执行来解决以下问题。")
        parts.append("")
        parts.append("【格式要求】")
        parts.append("每轮输出必须包含以下两种之一：")
        parts.append("1. 推理: [你的推理过程]")
        parts.append("2. 代码: ```python\n[Python代码]\n```")
        parts.append("")
        parts.append("当问题解决后，输出：")
        parts.append("答案：[最终结果]")
        parts.append("")

        if history:
            parts.append("【执行历史】")
            for h in history[-5:]:
                parts.append(f"- {h.get('action', '')}: {str(h.get('result', ''))[:100]}")
            parts.append("")

        parts.append(f"【题目】\n{problem}")
        return "\n".join(parts)

    def build_verification_prompt(self, problem: str, answer: str,
                                   reasoning: str = "") -> str:
        """v4.0: 构建验证器打分Prompt"""
        return f"""请作为数学答案验证器，评估以下解答是否正确。

【题目】
{problem}

【解答过程】
{reasoning[:500] if reasoning else "(无详细过程)"}

【给出的答案】
{answer}

请从以下维度打分(0-10分)：
- 答案正确性 (0-4分)
- 推理过程合理性 (0-3分)
- 计算步骤准确性 (0-3分)

请以以下格式输出：
总分: [0-10之间的数字]
评价: [简要说明]"""



# =============================================================================
# v4.0: 增强版AnswerParser - 支持Thinking Process输出格式
# =============================================================================

class AnswerParser:
    """从模型输出中提取并标准化最终答案 (v4.0: 增加Thinking Process支持)"""

    # v4.0: 增加更多匹配模式，特别是应对Thinking Process输出
    ANSWER_PATTERNS = [
        # 标准格式
        r'(?:答案|Answer|最终结果)\s*[：:]\s*(.+?)(?:\n|$)',
        r'\\boxed\{(.+?)\}',
        r'\\text\{答案\}\s*[：:]\s*(.+?)(?:\n|$)',
        # v4.0 NEW: 应对Thinking Process中的答案格式
        r'(?:Final Answer|最终答案)\s*[：:]\s*(.+?)(?:\n|$)',
        r'\*\*(?:Answer|答案)\*\*\s*[：:]?\s*(.+?)(?:\n|$)',
        r'【答案】\s*(.+?)(?:\n|$)',
        # v4.0 NEW: 英文常见格式
        r'The answer is\s*:?\s*(.+?)(?:\n|\.|$)',
        r'Ans(?:wer)?[.:\s]+(.+?)(?:\n|$)',
        # v4.0 NEW: 证明题结论
        r'(?:因此|所以|故|综上)[，,]\s*(.+?)(?:\n|$)',
        # v4.0 NEW: 从计算过程最后一行提取
        r'(?:=\s*)([^=\n]{1,50})(?:\n|$)',
    ]

    # v4.0 NEW: 选择题专用模式
    CHOICE_PATTERNS = [
        r'答案\s*[：:]\s*([A-D])',
        r'(?:选择|选)\s*[：:]?\s*([A-D])',
        r'(?:选项|答案)\s*[：:]?\s*([A-D])',
        r'\*\*([A-D])\*\*',  # 粗体字母
    ]

    def parse(self, text: str, question_type: str = "computation") -> str:
        """从文本中提取最终答案"""
        text = text.strip()
        if not text:
            return ""

        # 证明题特殊处理
        if question_type == "proof":
            conclusion = self._extract_proof_conclusion(text)
            if conclusion:
                return self._clean(conclusion)

        # 选择题特殊处理 (v4.0: 增强)
        if question_type == "choice":
            for pattern in self.CHOICE_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
            # 回退：查找独立的大写字母
            for line in reversed(text.split('\n')):
                line = line.strip()
                clean = re.sub(r'[\s`*_]', '', line)
                if clean in 'ABCD' and len(clean) == 1:
                    return clean

        # 填空题处理
        if question_type == "fill_blank":
            match = re.search(r'答案\s*[：:]\s*(.+?)(?:\n|$)', text)
            if match:
                return self._clean(match.group(1))

        # 通用模式匹配 (v4.0: 按优先级顺序)
        for pattern in self.ANSWER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                cleaned = self._clean(match.group(1))
                if cleaned and len(cleaned) < 500:
                    return cleaned

        # v4.0 NEW: 从Thinking Process中提取"Final Answer:"之后的内容
        final_answer_section = self._extract_from_thinking(text)
        if final_answer_section:
            return final_answer_section

        # 回退：提取最后几行的非空内容
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in reversed(lines[-8:]):
            if len(line) < 200 and not line.endswith('。'):
                cleaned = self._clean(line)
                if cleaned and len(cleaned) > 0:
                    return cleaned

        return text[-500:] if len(text) > 500 else text

    def _extract_from_thinking(self, text: str) -> Optional[str]:
        """v4.0 NEW: 从Thinking Process格式中提取答案"""
        # 查找"Format the Output"或"Final Answer"之后的最后一行
        markers = [
            r'Format the Output:\s*\n*\s*\*\*\*?\s*\n*(.+?)(?:\n|$)',
            r'Final Answer:\s*\*\*\*?\s*(.+?)(?:\n|$)',
            r'Answer:\s*\*\*\*?\s*(.+?)(?:\n|$)',
            r'\*\*Answer:\*\*\s*(.+?)(?:\n|$)',
        ]
        for pattern in markers:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
            if matches:
                # 取最后一个匹配
                return self._clean(matches[-1].group(1))
        return None

    def _extract_proof_conclusion(self, text: str) -> Optional[str]:
        """从证明文本中提取结论"""
        match = re.search(r'因此[，,]\s*(.+?)(?:\n|证毕|。\s*$)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'(.+?)(?:\n|^)证毕[。\.]', text, re.DOTALL)
        if match:
            lines = [l.strip() for l in match.group(1).split('\n') if l.strip()]
            if lines:
                return lines[-1]
        return None

    def _clean(self, s: str) -> str:
        """清理答案格式"""
        s = s.strip().strip('`').strip()
        s = re.sub(r'\[COMPUTE\s*[:：]\s*.+?\]\s*=>\s*', '', s)
        s = re.sub(r'^[=\s]+', '', s)
        return ' '.join(s.split())

    def normalize_latex(self, s: str) -> str:
        """标准化LaTeX格式"""
        s = s.replace('\u2212', '-')
        s = re.sub(r'\(-?\d+\)/\(-?\d+\)',
                   lambda m: m.group(0).replace('(', '').replace(')', ''), s)
        return s


# =============================================================================
# v4.0 NEW: ReAct Python执行器 (借鉴MathRouterAgent)
# =============================================================================

class ReActPythonExecutor:
    """ReAct风格的Python交互执行器 - 推理→执行→观察循环"""

    def __init__(self, llm_client, sympy_executor, config: AgentConfig):
        self.llm = llm_client
        self.sympy = sympy_executor
        self.config = config
        self.max_steps = config.react_max_steps
        self.timeout_per_step = config.react_timeout_per_step

    def solve(self, problem: str, analysis: Dict) -> Dict[str, Any]:
        """通过ReAct循环求解问题"""
        history = []
        prompt_engineer = PromptEngineer()

        for step in range(self.max_steps):
            # 构建prompt
            prompt = prompt_engineer.build_react_prompt(problem, analysis, history)

            # 调用LLM
            try:
                response = self.llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048
                )
            except Exception as e:
                return {"success": False, "error": f"LLM error: {e}", "answer": ""}

            # 检测是否已给出答案
            answer_match = re.search(r'(?:答案|Answer)\s*[：:]\s*(.+?)(?:\n|$)', response)
            if answer_match:
                return {
                    "success": True,
                    "answer": answer_match.group(1).strip(),
                    "steps": step + 1,
                    "history": history
                }

            # 提取Python代码
            code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
            if not code_blocks:
                code_blocks = re.findall(r'```\n(.*?)```', response, re.DOTALL)

            if code_blocks:
                code = code_blocks[-1].strip()
                # 幻觉检测
                hallucination = HallucinationDetector.detect(code)
                if hallucination["has_hallucination"]:
                    history.append({
                        "action": "detect_hallucination",
                        "result": f"Found {hallucination['count']} fake execution markers"
                    })

                # 执行代码
                result = self.sympy.execute(code)
                history.append({
                    "action": "execute_python",
                    "code": code[:200],
                    "result": str(result.get("result", "N/A"))[:200] if result["success"] else result.get("error", "Error"),
                    "success": result["success"]
                })
            else:
                # 纯推理步骤
                history.append({
                    "action": "reasoning",
                    "result": response[:300]
                })

        # 达到最大步数，从history中提取最佳答案
        return {
            "success": False,
            "error": "Max steps reached",
            "answer": self._extract_best_from_history(history),
            "history": history
        }

    def _extract_best_from_history(self, history: List[Dict]) -> str:
        """从历史记录中提取最佳答案"""
        # 查找成功的代码执行结果
        for entry in reversed(history):
            if entry.get("action") == "execute_python" and entry.get("success"):
                result = str(entry.get("result", ""))
                if result and result not in ("N/A", "None"):
                    return result
        return ""


# =============================================================================
# v4.0 NEW: LLM验证器打分 (借鉴MathRouterAgent)
# =============================================================================

class LLMVerifier:
    """让LLM作为评委对候选答案进行0-10分打分"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def score(self, problem: str, answer: str, reasoning: str = "",
              num_rounds: int = 2) -> Dict[str, Any]:
        """对答案进行打分，可多次打分取平均"""
        scores = []
        explanations = []

        prompt_engineer = PromptEngineer()

        for _ in range(num_rounds):
            prompt = prompt_engineer.build_verification_prompt(problem, answer, reasoning)
            try:
                response = self.llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.7,  # 稍高温度增加多样性
                    max_tokens=512
                )

                # 提取分数
                score_match = re.search(r'总分\s*[：:]\s*(\d+(?:\.\d+)?)', response)
                if score_match:
                    score = float(score_match.group(1))
                    score = max(0, min(10, score))
                    scores.append(score)
                    explanations.append(response[:200])
            except Exception:
                continue

        if not scores:
            return {"score": 5.0, "confidence": 0.0, "explanation": "Verification failed"}

        avg_score = sum(scores) / len(scores)
        confidence = 1.0 - (max(scores) - min(scores)) / 10.0 if len(scores) > 1 else 0.5

        return {
            "score": avg_score,
            "confidence": confidence,
            "explanation": explanations[0] if explanations else "",
            "all_scores": scores
        }

    def select_best(self, problem: str, candidates: List[Dict]) -> Dict:
        """从多个候选中选择最佳答案"""
        scored = []
        for cand in candidates:
            result = self.score(problem, cand.get("answer", ""),
                               cand.get("response", "")[:500])
            scored.append({
                **cand,
                "verifier_score": result["score"],
                "verifier_confidence": result["confidence"]
            })

        # 选择验证器分数最高的
        best = max(scored, key=lambda x: x.get("verifier_score", 0))
        return best


# =============================================================================
# v4.0 NEW: 答案回捞 (借鉴ICMA + MathRouterAgent)
# =============================================================================

class AnswerRecovery:
    """当答案提取失败时，从完整输出中回捞答案"""

    # v4.0: 各种可能的答案标记
    RECOVERY_PATTERNS = [
        # 数值答案 (最后一行是数字)
        r'(?:\n|^)\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$',
        # LaTeX分数
        r'\\frac\{([^}]+)\}\{([^}]+)\}',
        # 简洁表达式
        r'(?:=\s*)([\\a-zA-Z0-9\+\-\*/\^\(\)\{\}\[\]_\s]{1,80})(?:\n|$)',
        # 域表示
        r'\\mathbb\{[A-Z]\}_\{([^}]+)\}',
        # 纯数字 (选择题回退)
        r'\b(\d+)\b',
    ]

    @classmethod
    def recover(cls, text: str, question_type: str = "computation") -> Optional[str]:
        """从完整输出中尝试回捞答案"""
        if not text or len(text) < 5:
            return None

        # 策略1: 查找最后出现的数值
        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', text)
        if numbers and question_type in ("computation", "fill_blank"):
            # 返回最后一个合理的数值
            for num in reversed(numbers[-5:]):
                try:
                    val = float(num)
                    if abs(val) < 1e10:  # 排除异常大数
                        return num
                except ValueError:
                    continue

        # 策略2: 查找LaTeX表达式
        latex_exprs = re.findall(r'\$([^$]+)\$', text)
        if latex_exprs:
            # 返回最后一个较短的LaTeX表达式
            for expr in reversed(latex_exprs[-3:]):
                if 1 < len(expr) < 100:
                    return expr.strip()

        # 策略3: 匹配特定模式
        for pattern in cls.RECOVERY_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[-1], tuple):
                    return '/'.join(matches[-1])
                return str(matches[-1]).strip()

        return None


# =============================================================================
# 答案等价性判断 (v4.0: 增强矩阵和复数比较)
# =============================================================================

class AnswerEquivalence:
    """判断两个数学答案是否等价"""

    EPSILON = 1e-5
    MATRIX_EPSILON = 1e-6

    @classmethod
    def are_equivalent(cls, a: str, b: str, question_type: str = "computation") -> bool:
        if not a or not b:
            return False
        a, b = a.strip(), b.strip()
        if a == b:
            return True

        # 选择题直接比较
        if question_type == "choice":
            return a.upper() == b.upper()

        # 标准化后比较
        a_norm = cls._normalize(a)
        b_norm = cls._normalize(b)
        if a_norm == b_norm:
            return True

        # SymPy符号等价
        try:
            import sympy as sp
            a_expr = cls._to_sympy(a)
            b_expr = cls._to_sympy(b)
            if a_expr is not None and b_expr is not None:
                if cls._sympy_equiv(a_expr, b_expr):
                    return True
        except Exception:
            pass

        # 数值近似比较
        try:
            a_val = cls._to_float(a)
            b_val = cls._to_float(b)
            if a_val is not None and b_val is not None:
                if abs(a_val - b_val) < cls.EPSILON * max(abs(a_val), abs(b_val), 1.0):
                    return True
        except Exception:
            pass

        # 字符串相似度
        if cls._string_similarity(a_norm, b_norm) > 0.9:
            return True

        return False

    @classmethod
    def _normalize(cls, s: str) -> str:
        s = s.strip().lower()
        s = s.replace('\\', '')
        s = s.replace(' ', '')
        s = s.replace('\u2212', '-')
        s = re.sub(r'\$+', '', s)
        s = re.sub(r'\^{?\\circ}?', 'deg', s)
        return s

    @classmethod
    def _to_sympy(cls, s: str):
        import sympy as sp
        try:
            replacements = {
                'pi': str(sp.pi), 'e': str(sp.E),
                'i': str(sp.I), 'I': str(sp.I),
                'oo': str(sp.oo), 'infty': str(sp.oo),
                '\u221a': 'sqrt', '\u221e': 'oo',
            }
            expr_str = s
            for old, new in replacements.items():
                expr_str = expr_str.replace(old, new)
            return sp.sympify(expr_str)
        except Exception:
            return None

    @classmethod
    def _sympy_equiv(cls, a, b) -> bool:
        import sympy as sp
        try:
            if a == b:
                return True
            if sp.simplify(a - b) == 0:
                return True
            if hasattr(a, 'equals') and a.equals(b):
                return True
            # 矩阵比较
            if isinstance(a, sp.Matrix) and isinstance(b, sp.Matrix):
                if a.shape == b.shape:
                    return all(abs(a[i,j] - b[i,j]) < cls.MATRIX_EPSILON
                              for i in range(a.rows) for j in range(a.cols))
            return False
        except Exception:
            return False

    @classmethod
    def _to_float(cls, s: str) -> Optional[float]:
        try:
            import sympy as sp
            expr = cls._to_sympy(s)
            if expr is not None:
                val = float(expr.evalf())
                if val == val:  # 排除NaN
                    return val
        except Exception:
            pass
        try:
            cleaned = re.sub(r'[^0-9.\-+eE]', '', s)
            return float(cleaned)
        except Exception:
            return None

    @classmethod
    def _string_similarity(cls, a: str, b: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()


# =============================================================================
# 并查集 + 多候选投票
# =============================================================================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def cluster_and_vote(answers: List[str], question_type: str = "computation") -> List[Dict]:
    """并查集聚类 + 多数投票"""
    n = len(answers)
    if n == 0:
        return []
    if n == 1:
        return [{"representative": answers[0], "members": answers, "score": 1.0}]

    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if AnswerEquivalence.are_equivalent(answers[i], answers[j], question_type):
                uf.union(i, j)

    groups = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    clusters = []
    for indices in groups.values():
        members = [answers[i] for i in indices]
        rep = Counter(members).most_common(1)[0][0]
        score = len(indices) / n
        clusters.append({"representative": rep, "members": members, "score": score})

    return sorted(clusters, key=lambda x: x["score"], reverse=True)



# =============================================================================
# v4.0: ReasoningAgent主类 - 整合所有新功能
# =============================================================================

class ReasoningAgent:
    """数学推理智能体主类 (v4.0全面优化版)"""

    def __init__(self, client, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.client = LLMClient(client)

        # 自动检测模型类型并适配配置
        detected_model = self.client.detect_model()
        self.config.adapt_for_model(detected_model)

        # 初始化子组件
        self.analyzer = ProblemAnalyzer()
        self.prompt_engineer = PromptEngineer()
        self.answer_parser = AnswerParser()
        self.sympy_executor = SafeSymPyExecutor(self.config.sympy_timeout)

        # v4.0 NEW: 初始化新组件
        self.react_executor = ReActPythonExecutor(
            self.client, self.sympy_executor, self.config
        ) if self.config.enable_react_python else None
        self.verifier = LLMVerifier(self.client) if self.config.enable_verifier_scoring else None
        self.token_budget = TokenBudget() if self.config.enable_token_budget else None

        # 状态跟踪
        self.trace_log: List[Dict] = []
        self.call_count = 0
        self.start_time = None
        self._call_timestamps: List[float] = []

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def solve(self, problem: str, metadata: Dict) -> Dict:
        """主求解入口
        Returns: {"final_response": str, "trace": list}
        """
        self.start_time = time.time()
        self.trace_log = []
        self.call_count = 0
        self._call_timestamps = []

        try:
            with TimeoutContext(self.config.problem_timeout - 5):
                return self._solve_internal(problem, metadata)
        except TimeoutError:
            return self._fallback_response(problem, "timeout")
        except Exception as e:
            return self._fallback_response(problem, str(e))

    def _solve_internal(self, problem: str, metadata: Dict) -> Dict:
        """内部求解逻辑 (v4.0: 多策略融合)"""
        all_candidates = []

        # Step 1: 问题分析
        analysis = self.analyzer.analyze(problem)
        self._log("analysis", analysis)

        # v4.0: Token预算检查
        if self.token_budget:
            self.token_budget.consume("analysis_call")

        # Step 2: 确定策略
        difficulty = analysis.get("difficulty", "medium")
        qt = analysis.get("question_type", "computation")
        num_calls = self._mode_to_calls(difficulty)

        self._log("strategy", {
            "thinking_mode": self.config.thinking_mode,
            "difficulty": difficulty,
            "num_candidates": num_calls,
            "question_type": qt,
            "subdomain": analysis.get("subdomain"),
            "model_type": self.config.model_type,
        })

        # Step 3: 多候选生成 (v4.0: 增加ReAct Python执行)
        # 策略A: 标准多候选生成
        standard_candidates = self._generate_candidates(problem, analysis, num_calls)
        all_candidates.extend(standard_candidates)

        # v4.0 NEW: 策略B: ReAct Python执行 (如果启用且预算充足)
        if (self.react_executor and self.config.enable_react_python and
            (not self.token_budget or self.token_budget.can_afford("candidate_generation", 1))):
            try:
                react_result = self.react_executor.solve(problem, analysis)
                if react_result.get("success") and react_result.get("answer"):
                    all_candidates.append({
                        "answer": react_result["answer"],
                        "response": f"ReAct({react_result['steps']} steps)",
                        "source": "react",
                        "sympy_verified": True,
                    })
                    self._log("react_execution", {
                        "steps": react_result.get("steps", 0),
                        "answer": react_result["answer"],
                    })
            except Exception as e:
                self._log("react_error", {"error": str(e)})

        self._log("candidates", {
            "count": len(all_candidates),
            "answers": [c["answer"] for c in all_candidates],
            "sources": [c.get("source", "standard") for c in all_candidates],
        })

        # Step 4: 选择最佳答案
        best = self._select_best(all_candidates, problem, analysis)

        # v4.0 NEW: 答案契约检查
        if self.config.enable_answer_contract:
            contract = AnswerContract.analyze(problem, best.get("answer", ""))
            self._log("answer_contract", contract)
            # 如果契约检查失败，尝试从其他候选中找满足契约的答案
            if not contract["passed"] and len(all_candidates) > 1:
                best = self._find_contract_satisfying(problem, all_candidates, contract)

        self._log("selection", {
            "best_answer": best["answer"],
            "confidence": best.get("confidence", 0),
            "clusters": best.get("clusters", []),
        })

        # Step 5: 格式化答案
        final_answer = self._format_final_answer(best["answer"], analysis)

        # v4.0 NEW: 如果答案为空或无效，尝试回捞
        if (not final_answer or final_answer in ("XXX`.", "[最终结果]`.", "")) and \
           self.config.enable_answer_recovery:
            recovered = self._recover_answer(problem, all_candidates, best)
            if recovered:
                final_answer = recovered
                self._log("answer_recovery", {"recovered": recovered})

        # Step 6: 构建trace
        self._build_trace(problem, analysis, all_candidates, best)

        elapsed = time.time() - self.start_time
        self._log("completion", {
            "elapsed_seconds": elapsed,
            "total_llm_calls": self.call_count,
            "sympy_stats": self.sympy_executor.stats,
            "token_budget": self.token_budget.get_summary() if self.token_budget else None,
        })

        return {"final_response": final_answer, "trace": self.trace_log}

    # ------------------------------------------------------------------
    # v4.0: 候选生成 (增加Thinking Process应对)
    # ------------------------------------------------------------------

    def _generate_candidates(self, problem: str, analysis: Dict,
                             num_calls: int) -> List[Dict]:
        """生成多个候选答案"""
        candidates = []
        prompt = self.prompt_engineer.build_prompt(problem, analysis)

        # v4.0: 温度多样化策略
        temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        # v4.0: Token预算检查
        if self.token_budget and not self.token_budget.can_afford("candidate_generation", num_calls):
            num_calls = max(1, num_calls // 2)
            self._log("token_budget_cut", {"reduced_to": num_calls})

        for i in range(num_calls):
            temp = temperatures[i % len(temperatures)]
            try:
                raw = self._call_llm(prompt, temperature=temp,
                                     max_tokens=self.config.max_tokens)

                # v4.0: 幻觉检测
                if self.config.enable_hallucination_detection:
                    hallucination = HallucinationDetector.detect(raw)
                    if hallucination["has_hallucination"]:
                        self._log("hallucination_detected", {
                            "candidate_idx": i,
                            "count": hallucination["count"]
                        })

                # 提取答案
                answer = self.answer_parser.parse(raw, analysis.get("question_type", "computation"))

                # v4.0: 如果提取失败，尝试回捞
                if not answer or answer in ("XXX`.", "[最终结果]`."):
                    recovered = AnswerRecovery.recover(raw, analysis.get("question_type"))
                    if recovered:
                        answer = recovered

                # 执行COMPUTE标记
                compute_results, sympy_verified = self._execute_computations(raw)

                candidates.append({
                    "answer": answer,
                    "response": raw,
                    "temperature": temp,
                    "compute_results": compute_results,
                    "sympy_verified": sympy_verified,
                    "source": "standard",
                })

                # 记录token消耗
                if self.token_budget:
                    self.token_budget.consume("candidate_generation")

            except Exception as e:
                self._log("candidate_error", {"idx": i, "error": str(e)})
                continue

        return candidates

    def _execute_computations(self, text: str) -> Tuple[List[Dict], bool]:
        """执行文本中的[COMPUTE: ...]标记"""
        computations = re.findall(r'\[COMPUTE\s*[:：]\s*([^\]]+?)\s*\]', text)
        results = []
        all_success = True
        for expr in computations:
            result = self.sympy_executor.execute(expr)
            results.append({"expr": expr, **result})
            if not result["success"]:
                all_success = False
        return results, all_success and len(results) > 0

    # ------------------------------------------------------------------
    # v4.0: 答案选择 (并查集投票 + 验证器打分融合)
    # ------------------------------------------------------------------

    def _select_best(self, candidates: List[Dict], problem: str,
                     analysis: Dict) -> Dict:
        """选择最佳答案"""
        if not candidates:
            return {"answer": "", "confidence": 0, "clusters": []}
        if len(candidates) == 1:
            return {**candidates[0], "confidence": 0.5, "clusters": []}

        qt = analysis.get("question_type", "computation")
        answers = [c["answer"] for c in candidates]

        # 方法1: 并查集聚类投票
        clusters = cluster_and_vote(answers, qt)

        # 方法2: v4.0 NEW - 验证器打分 (如果启用)
        if self.verifier and self.config.enable_verifier_scoring:
            try:
                verifier_scores = {}
                for i, cand in enumerate(candidates):
                    score_result = self.verifier.score(
                        problem, cand["answer"],
                        cand.get("response", "")[:500],
                        num_rounds=1  # 节省token
                    )
                    verifier_scores[i] = score_result["score"]

                # 融合投票和验证器分数
                best_cluster = clusters[0]
                best_idx = None
                best_fusion_score = -1

                for i, cand in enumerate(candidates):
                    vote_score = next((c["score"] for c in clusters
                                      if AnswerEquivalence.are_equivalent(
                                          c["representative"], cand["answer"], qt)), 0)
                    v_score = verifier_scores.get(i, 5.0) / 10.0
                    fusion = 0.6 * vote_score + 0.4 * v_score
                    if fusion > best_fusion_score:
                        best_fusion_score = fusion
                        best_idx = i

                if best_idx is not None:
                    return {
                        **candidates[best_idx],
                        "confidence": best_fusion_score,
                        "clusters": clusters,
                        "vote_score": vote_score,
                        "verifier_score": verifier_scores.get(best_idx, 0),
                    }
            except Exception as e:
                self._log("verifier_error", {"error": str(e)})

        # 回退: 纯投票
        best_cluster = clusters[0]
        best_candidate = next(
            (c for c in candidates
             if AnswerEquivalence.are_equivalent(c["answer"], best_cluster["representative"], qt)),
            candidates[0]
        )
        return {
            **best_candidate,
            "confidence": best_cluster["score"],
            "clusters": clusters,
        }

    # ------------------------------------------------------------------
    # v4.0 NEW: 找到满足契约的答案
    # ------------------------------------------------------------------

    def _find_contract_satisfying(self, problem: str, candidates: List[Dict],
                                   contract: Dict) -> Dict:
        """从候选中找到满足答案契约的答案"""
        for cand in candidates:
            check = AnswerContract.analyze(problem, cand.get("answer", ""))
            if check["passed"]:
                return cand
        return candidates[0]  # 回退到第一个

    # ------------------------------------------------------------------
    # v4.0 NEW: 答案回捞
    # ------------------------------------------------------------------

    def _recover_answer(self, problem: str, candidates: List[Dict],
                        best: Dict) -> Optional[str]:
        """尝试从各种来源回捞答案"""
        # 从最佳候选的完整响应中回捞
        response = best.get("response", "")
        if response:
            recovered = AnswerRecovery.recover(response, "computation")
            if recovered:
                return recovered

        # 从所有候选中回捞
        for cand in candidates:
            response = cand.get("response", "")
            if response:
                recovered = AnswerRecovery.recover(response, "computation")
                if recovered:
                    return recovered

        # 从计算结果中回捞
        for cand in candidates:
            compute_results = cand.get("compute_results", [])
            for cr in compute_results:
                if cr.get("success") and cr.get("result") is not None:
                    return str(cr["result"])

        return None

    # ------------------------------------------------------------------
    # 答案格式化
    # ------------------------------------------------------------------

    def _format_final_answer(self, answer: str, analysis: Dict) -> str:
        """格式化最终答案"""
        if not answer:
            return ""
        answer = answer.strip()
        answer = self.answer_parser.normalize_latex(answer)

        if analysis.get("question_type") == "choice":
            answer = answer.strip().upper()
            if answer and answer[0] in 'ABCD':
                return answer[0]

        if analysis.get("question_type") == "fill_blank":
            answer = re.sub(r'^(?:答案|结果为|等于)\s*[：:]\s*', '', answer)

        return answer

    # ------------------------------------------------------------------
    # Trace构建
    # ------------------------------------------------------------------

    def _build_trace(self, problem: str, analysis: Dict,
                     candidates: List[Dict], best: Dict):
        """构建高质量trace"""
        trace = [{
            "type": "analysis",
            "subdomain": analysis.get("subdomain"),
            "subdomain_scores": analysis.get("subdomain_scores", {}),
            "question_type": analysis.get("question_type"),
            "difficulty": analysis.get("difficulty"),
            "key_concepts": analysis.get("key_concepts", []),
        }]

        if analysis.get("rag_context"):
            trace.append({
                "type": "knowledge_retrieval",
                "context": analysis["rag_context"][:500],
            })

        # 记录推理过程
        best_response = best.get("response", "")
        if best_response:
            reasoning_lines = [l.strip() for l in best_response.split('\n')
                              if l.strip() and not l.strip().startswith('答案')]
            trace.append({"type": "reasoning", "steps": reasoning_lines[:30]})

        # 记录验证
        best_candidate = next((c for c in candidates if c["answer"] == best["answer"]), None)
        if best_candidate and best_candidate.get("compute_results"):
            trace.append({
                "type": "verification",
                "sympy_results": best_candidate["compute_results"],
                "sympy_verified": best_candidate.get("sympy_verified"),
            })

        # 记录选择
        trace.append({
            "type": "selection",
            "final_answer": best["answer"],
            "confidence": best.get("confidence", 0),
            "vote_ratio": f"{best.get('vote_count', 0)}/{len(candidates)}",
            "clusters": best.get("clusters", []),
        })

        # 合并到trace_log
        self.trace_log.extend([{"step": "trace", **t} for t in trace])

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _log(self, step: str, data: Dict):
        """记录trace"""
        now = time.time()
        elapsed = now - self.start_time if self.start_time else 0.0
        self.trace_log.append({
            "step": step,
            "timestamp": now,
            "elapsed_seconds": elapsed,
            **data
        })

    def _call_llm(self, prompt: str, temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None) -> str:
        """调用LLM，带重试和RPM流控"""
        temp = temperature if temperature is not None else self.config.policy_temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        messages = [{"role": "user", "content": prompt}]

        # v4.0: 上下文裁剪
        if self.config.enable_context_pruning:
            messages = ContextPruner.prune(messages)

        kwargs = {}
        if self.config.enable_thinking:
            tm = self._get_thinking_mode()
            if tm:
                kwargs["enable_thinking"] = True

        # RPM流控
        self._enforce_rpm_limit()

        # 指数退避重试
        max_retries = self.config.max_llm_retries
        base_delay = self.config.retry_base_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                self._call_timestamps.append(time.time())
                result = self.client.chat(messages, temperature=temp, max_tokens=tokens, **kwargs)
                self.call_count += 1
                return result
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                if any(kw in err_str for kw in ['rate limit', 'too many', '429']):
                    time.sleep(base_delay * (4 ** attempt))
                elif attempt == max_retries - 1:
                    raise
                else:
                    time.sleep(base_delay * (2 ** attempt))

        raise last_exception

    def _enforce_rpm_limit(self):
        """RPM流控"""
        limit = self.config.rpm_limit
        if limit <= 0:
            return
        window = 60.0
        now = time.time()
        self._call_timestamps = [t for t in self._call_timestamps if now - t < window]
        if len(self._call_timestamps) >= limit:
            sleep_time = window - (now - self._call_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self._enforce_rpm_limit()

    def _get_thinking_mode(self) -> Optional[str]:
        mode_map = {"quick": "light", "standard": "medium", "deep": "heavy"}
        return mode_map.get(self.config.thinking_mode)

    def _mode_to_calls(self, difficulty: str) -> int:
        """根据模式和难度确定候选数"""
        mode = self.config.thinking_mode
        # v4.0: s2模型减少候选数以节省token
        if self.config.model_type == "s2":
            if mode == "quick":
                return 2
            elif mode == "deep":
                return 5 if difficulty == "hard" else 3
            return 4 if difficulty == "hard" else 3
        else:
            if mode == "quick":
                return 3 if difficulty == "hard" else 2
            elif mode == "deep":
                return 8 if difficulty == "hard" else 5
            return 6 if difficulty == "hard" else 4

    def _fallback_response(self, problem: str, reason: str) -> Dict:
        """超时/异常回退"""
        fallback_answer = ""
        for entry in self.trace_log:
            if entry.get("step") == "candidates":
                answers = entry.get("answers", [])
                for a in answers:
                    if a and a.strip() and a not in ("XXX`.", "[最终结果]`."):
                        fallback_answer = a.strip()
                        break
                break

        return {
            "final_response": fallback_answer,
            "trace": self.trace_log + [{"type": "fallback", "reason": reason, "answer": fallback_answer}],
        }
