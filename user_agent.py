# -*- coding: utf-8 -*-
"""
================================================================================
 基于 Intern-S 系列大模型的数学推理智能体
 2026年度中国青年科技创新"揭榜挂帅"擂台赛 · 书生赛道
================================================================================

设计目标（评分核心依据 = final_response 答案正确率）：

  1. 多候选生成 + Self-Consistency 自洽投票
     - 多温度采样生成 N 条独立解答
     - 答案归一化 + 符号级等价聚类（Union-Find），多数表决选出最自洽答案

  2. 强鲁棒的答案抽取与格式化（AnswerParser / AnswerFormatter）
     - 覆盖 "答案：X" / \\boxed{} / **Answer:** / Final Answer / 因此所以 / 末行数值
     - 针对 intern-s2-preview 的 Thinking Process 截断输出做末位回捞
     - 选择题字母、判断题关键词、数值/表达式的分类型格式化

  3. SymPy 符号验证（双通道）
     - 候选答案之间的符号等价性判断（simplify(a-b)==0，带超时保护）
     - LLM 生成 sympy 计算脚本 -> 受限沙箱执行 -> 计算结果作为独立票源参与投票

  4. 反思纠错（Reflection）
     - 当投票无多数 / 答案契约校验失败 / 抽取失败比例过高时，
       匿名展示分歧答案，引导模型重新推导并自我纠错

  5. 工程稳定性
     - 适配 intern-s2-preview-397b：解析 choices/message/reasoning_content，
       enable_thinking=False 自适应（官方 client 不支持时自动降级并容忍长思考）
     - 指数退避重试、墙钟时间预算、并发候选生成、异常全兜底
     - final_response 保证非空字符串，返回值保证 JSON 可序列化

接口规范（平台强制，不可修改）：
    class ReasoningAgent:
        def __init__(self, client, *args, **kwargs)
        def solve(self, problem: str, metadata: dict) -> dict
        # 返回 {"final_response": str, "trace": list}

外部依赖：仅 sympy>=1.12（其余全部为 Python 标准库）
================================================================================
"""

import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

try:
    import sympy
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    _SYMPY_OK = True
except Exception:  # pragma: no cover - sympy 不可用时整体降级
    sympy = None
    _SYMPY_OK = False


# ==============================================================================
# 配置
# ==============================================================================

@dataclass
class AgentConfig:
    """用户可控参数。所有参数均有安全默认值，不配置也可直接运行。"""

    # ---- 候选生成（self-consistency 规模）----
    num_candidates: int = 4                 # standard 模式候选数
    candidate_temperatures: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
    max_candidates_hard_cap: int = 6
    gen_workers: int = 4                    # 候选生成并发数（=默认候选数，一波发完）

    # ---- 反思纠错 ----
    enable_reflection: bool = True
    max_reflect_rounds: int = 1             # 反思轮数（每轮补充 2 个候选）
    reflect_temperature: float = 0.3

    # ---- SymPy 符号验证 ----
    enable_equivalence: bool = True         # 候选间符号等价聚类
    enable_code_verification: bool = True   # LLM 生成 sympy 脚本独立验算
    code_exec_timeout: float = 12.0         # 沙箱执行超时（秒）
    sympy_simplify_timeout: float = 4.0     # 单条等价判断超时（秒）
    code_vote_weight: int = 2               # 代码验算结果在投票中的权重

    # ---- LLM 裁判（仅平票时启用，控制成本）----
    enable_judge_on_tie: bool = True
    judge_temperature: float = 0.0

    # ---- 调用控制 ----
    # max_tokens=8192：s2 在无法关闭 thinking 时会把思考写进 content，
    # 4096 会在难题上截断（实测 4847 token 才完成），8192 保证答案输出完整。
    max_tokens: int = 8192
    code_max_tokens: int = 4096
    max_retries: int = 3                    # 单次 LLM 调用重试次数
    retry_base_sleep: float = 2.0
    disable_thinking: bool = True           # 尝试关闭 s2 的 Thinking Process
    time_budget_sec: float = 480.0          # 单题墙钟预算（超时则跳过低优先级阶段）

    # ---- 输出 ----
    max_answer_chars: int = 200             # final_response 长度护栏
    trace_candidate_chars: int = 600        # trace 中候选原文截断长度


# ==============================================================================
# 通用工具
# ==============================================================================

def run_with_timeout(fn, timeout: float, default=None):
    """在守护线程中执行 fn，超时返回 default。线程泄漏可接受（任务很小）。"""
    box = {"value": default}

    def _target():
        try:
            box["value"] = fn()
        except Exception:
            box["value"] = default

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(max(timeout, 0.05))
    return box["value"]


def json_safe(obj: Any) -> Any:
    """保证对象可 JSON 序列化。"""
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {str(k): json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_safe(v) for v in obj]
        return str(obj)


def clip(text: Any, limit: int) -> str:
    s = str(text) if text is not None else ""
    return s if len(s) <= limit else s[:limit] + "…"


# ==============================================================================
# LLM 调用封装：anti-thinking 自适应 + 重试 + 统计
# ==============================================================================

class LLMCaller:
    """
    只依赖公开约定的 client.chat(messages, temperature, max_tokens) 接口。
    - 优先尝试 enable_thinking=False（对 intern-s2-preview 至关重要：
      实测可避免 Thinking Process 耗尽 4096 token 导致答案被截断）；
      若官方 client 不支持该参数（TypeError），永久降级为标准调用。
    - 指数退避重试，统计调用次数供 trace 使用。
    """

    def __init__(self, client: Any, config: AgentConfig):
        self.client = client
        self.config = config
        self.calls = 0
        self.errors = 0
        self._thinking_kw_state = None  # None=未知 True=支持 False=不支持
        self._lock = threading.Lock()

    def _raw_chat(self, messages, temperature, max_tokens):
        kwargs = {}
        if self.config.disable_thinking and self._thinking_kw_state is not False:
            kwargs["enable_thinking"] = False
        if kwargs:
            try:
                resp = self.client.chat(
                    messages=messages, temperature=temperature,
                    max_tokens=max_tokens, **kwargs,
                )
                self._thinking_kw_state = True
                return resp
            except TypeError:
                # 官方 client 不接受额外参数 -> 永久降级
                self._thinking_kw_state = False
        return self.client.chat(
            messages=messages, temperature=temperature, max_tokens=max_tokens,
        )

    @staticmethod
    def _coerce_text(resp: Any) -> str:
        """把各种 client 返回格式统一成纯文本。

        适配 intern-s2-preview-397b：思考内容可能在 reasoning_content 字段，
        最终答案在 content 字段。优先取 content；content 为空时回退到
        reasoning_content 末尾（供 AnswerParser 截断回捞使用）。
        """
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            # OpenAI 风格：{"choices": [{"message": {"content": ..., "reasoning_content": ...}}]}
            choices = resp.get("choices")
            if isinstance(choices, list) and choices:
                try:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                    reasoning = msg.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning.strip():
                        return reasoning[-6000:]
                    delta = choices[0].get("delta") or {}
                    for k in ("content", "reasoning_content"):
                        v = delta.get(k)
                        if isinstance(v, str) and v.strip():
                            return v[-6000:]
                except (AttributeError, IndexError, TypeError):
                    pass
            for key in ("content", "text", "message"):
                v = resp.get(key)
                if isinstance(v, str):
                    return v
                if isinstance(v, dict):
                    c = v.get("content")
                    if isinstance(c, str) and c.strip():
                        return c
                    r = v.get("reasoning_content")
                    if isinstance(r, str) and r.strip():
                        return r[-6000:]
            # content 为空但顶层有 reasoning_content
            r = resp.get("reasoning_content")
            if isinstance(r, str) and r.strip():
                return r[-6000:]
        return str(resp) if resp is not None else ""

    def chat(self, messages, temperature=0.2, max_tokens=None) -> str:
        max_tokens = max_tokens or self.config.max_tokens
        last_err = None
        for attempt in range(self.config.max_retries):
            try:
                text = self._coerce_text(self._raw_chat(messages, temperature, max_tokens))
                with self._lock:
                    self.calls += 1
                return text
            except Exception as exc:  # noqa: BLE001 - 单题内任何调用异常都重试兜底
                last_err = exc
                with self._lock:
                    self.errors += 1
                if attempt + 1 < self.config.max_retries:
                    time.sleep(self.config.retry_base_sleep * (2 ** attempt)
                               + random.uniform(0, 0.8))
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} attempts: {last_err}")


# ==============================================================================
# 题目分析：题型 / 领域 / 期望答案形态
# ==============================================================================

class ProblemAnalyzer:
    """纯本地规则分析，零 LLM 成本。结果用于 prompt 定制、答案契约与解析优先级。"""

    DOMAIN_KEYWORDS = {
        "抽象代数": ["有限域", "mathbb{F}", "F_{", "GF(", " Galois", "群", "环", "域",
                     "同态", "同构", "置换", "Sylow", "理想"],
        "数论": ["素数", "质数", "整除", "同余", "mod ", "≡", "最大公约", "最小公倍",
                 "欧拉函数", "阶乘", "remainder", "modulo"],
        "组合数学": ["多少种", "选法", "排列", "组合", "C(", "binom", "种方法",
                     "分配", "染色", "鸽笼"],
        "概率统计": ["概率", "期望", "方差", "分布", "随机", "掷", "抽", "假设检验",
                     "置信区间", "显著性", "拒绝", "H_0", "H0", "均值"],
        "微积分": ["积分", "导数", "极限", "lim", "∫", "级数", "收敛", "发散",
                   " Taylor", "麦克劳林", "极值", "最值"],
        "线性代数": ["矩阵", "行列式", "特征值", "特征向量", "秩", "线性无关",
                     "eigenvalue", "determinant"],
        "微分方程": ["微分方程", "y'", "dy/dx", "ODE", "初值"],
        "几何": ["三角形", "圆", "面积", "体积", "角", "边长", "坐标", "距离"],
        "复分析": ["复数", "虚部", "实部", "模", "i)^", "z ="],
    }

    RE_CHOICE_BLOCK = re.compile(
        r"(?:^|[\s\n(（])[A-E][.、)）]\s*\S", re.MULTILINE)
    RE_PROOF = re.compile(r"证明|求证|prove|show that", re.IGNORECASE)
    RE_JUDGE = re.compile(r"是否|判断|对不对|真假|敛散性|是否拒绝")

    @classmethod
    def analyze(cls, problem: str) -> Dict[str, Any]:
        text = problem or ""
        is_choice = len(cls.RE_CHOICE_BLOCK.findall(text)) >= 2
        is_proof = bool(cls.RE_PROOF.search(text)) and not is_choice
        is_judge = bool(cls.RE_JUDGE.search(text)) and not is_choice

        domains = [d for d, kws in cls.DOMAIN_KEYWORDS.items()
                   if any(k in text for k in kws)]
        domain = domains[0] if domains else "通用数学"

        if is_choice:
            answer_kind = "letter"
        elif is_proof:
            answer_kind = "proof"
        elif re.search(r"敛散|收敛|发散|converge|diverge", text, re.IGNORECASE):
            answer_kind = "convergence"
        elif re.search(r"拒绝|假设检验|显著", text):
            answer_kind = "hypothesis"
        elif re.search(r"区间|confidence interval", text, re.IGNORECASE):
            answer_kind = "interval"
        elif is_judge:
            answer_kind = "judgment"
        else:
            answer_kind = "value"

        return {
            "is_choice": is_choice,
            "is_proof": is_proof,
            "answer_kind": answer_kind,
            "domain": domain,
            "length": len(text),
            # 一题多解信号：题目明确要求所有解/多选
            "wants_all": bool(re.search(
                r"所有|全部|解集|各个|哪些|所有根|所有值|一切", text)),
            "multi_select": bool(re.search(
                r"多选|哪些(?:选项|说法|结论)|下列.*(?:正确|错误).*(?:有|是)", text)),
        }


# ==============================================================================
# Prompt 工程
# ==============================================================================

class PromptEngineer:
    """分题型求解 prompt；核心：强制末行 '答案：X' 契约 + 抑制冗长思考。"""

    SYSTEM = (
        "You are a world-class mathematician. Solve the problem rigorously but concisely. "
        "Do NOT reveal internal deliberation. Keep the derivation short and clean."
    )

    CONTRACT_ZH = (
        "【输出格式 - 必须严格遵守】\n"
        "1. 给出简洁的关键推导（不超过8行），不要输出任何内心思考过程。\n"
        "2. 最后一行必须单独写：答案：<最终答案>\n"
        "3. <最终答案>中不要出现单位、解释文字或多余符号。"
    )

    KIND_HINT = {
        "letter": "本题为选择题，<最终答案>只能是选项字母本身（如 A），不要写选项内容。",
        "proof": "本题为证明题，先给出完整证明，最后一行写：答案：<一句话结论>。",
        "convergence": "<最终答案>只能是“收敛”或“发散”（可附一个关键词，如 条件收敛）。",
        "hypothesis": "<最终答案>必须以“拒绝 H0”或“不拒绝 H0”开头，可附检验统计量数值。",
        "interval": "<最终答案>写成闭/开区间形式，如 (a, b)，端点保留必要精度。",
        "judgment": "<最终答案>先写明确判断（是/否/正确/错误），再附关键数值。",
        "value": "<最终答案>为最简数值或最简表达式（分数优先于小数）。",
    }

    @classmethod
    def solver_messages(cls, problem: str, analysis: Dict[str, Any],
                        extra_hint: str = "") -> List[Dict[str, str]]:
        kind = analysis.get("answer_kind", "value")
        hint = cls.KIND_HINT.get(kind, cls.KIND_HINT["value"])
        if kind == "letter" and analysis.get("multi_select"):
            hint = ("本题为多选题，<最终答案>按字母升序写全部正确选项（如 AC），"
                    "不要写选项内容。")
        elif kind == "value" and analysis.get("wants_all"):
            hint = ("<最终答案>必须包含全部满足条件的值（如解集写成 2, 3 或 x=±2），"
                    "不要只写其中一个。")
        parts = [
            f"请解决下面的数学问题（领域：{analysis.get('domain', '通用数学')}）。",
            "",
            f"【题目】\n{problem}",
            "",
            cls.CONTRACT_ZH,
            hint,
        ]
        if extra_hint:
            parts += ["", f"【附加要求】\n{extra_hint}"]
        return [
            {"role": "system", "content": cls.SYSTEM},
            {"role": "user", "content": "\n".join(parts)},
        ]

    @classmethod
    def reflection_messages(cls, problem: str, analysis: Dict[str, Any],
                            distinct_answers: List[str]) -> List[Dict[str, str]]:
        shown = "\n".join(f"- {a}" for a in distinct_answers[:5]) or "（无有效答案）"
        hint = (
            "此前多次独立求解得到了以下互相分歧的答案：\n"
            f"{shown}\n"
            "请重新审视题目，逐步核查每一步计算，找出最可能正确的答案。"
            "不要盲从上面的任何答案，以你自己的严格推导为准。"
        )
        return cls.solver_messages(problem, analysis, extra_hint=hint)

    @classmethod
    def code_messages(cls, problem: str) -> List[Dict[str, str]]:
        user = (
            "为下面的数学题写一段 Python 代码来独立计算最终答案。\n"
            "可直接使用已导入的 sympy（记为 sp）与 math。只输出一个 ```python 代码块，不要解释。\n"
            "要求：\n"
            "1. 代码必须能直接运行，最后一行用 print(ANSWER) 输出最终答案；\n"
            "2. ANSWER 尽量是最简精确形式（分数/根式优先）；\n"
            "3. 不要使用 input、文件或网络。\n\n"
            f"【题目】\n{problem}"
        )
        return [
            {"role": "system", "content": "You are an expert in symbolic computing with sympy."},
            {"role": "user", "content": user},
        ]

    @classmethod
    def judge_messages(cls, problem: str, answers: List[str]) -> List[Dict[str, str]]:
        opts = "\n".join(f"{chr(65 + i)}. {a}" for i, a in enumerate(answers))
        user = (
            "下面是同一道数学题的若干个候选最终答案，它们互不相同。\n"
            "请独立重新解这道题，判断哪个候选答案正确。\n\n"
            f"【题目】\n{problem}\n\n【候选答案】\n{opts}\n\n"
            f"只输出一行：VERDICT: <字母>（字母范围 A 到 {chr(64 + len(answers))}）。"
        )
        return [
            {"role": "system", "content": "You are a strict math grader. Re-derive, then judge."},
            {"role": "user", "content": user},
        ]


# ==============================================================================
# 答案抽取：多模式 + 截断回捞
# ==============================================================================

class AnswerParser:
    """从模型自由文本中抽取最终答案。返回 (answer, method)；失败返回 ("", "none")。"""

    RE_ANSWER_ZH = re.compile(r"答案\s*[:：]\s*(.+)")
    RE_ANSWER_EN = re.compile(
        r"(?:final\s+answer|answer|ans)\s*(?:is\s*)?[:：]?\s*(.+)", re.IGNORECASE)
    RE_THE_ANSWER_IS = re.compile(r"the\s+answer\s+is\s+(.+)", re.IGNORECASE)
    RE_THEREFORE = re.compile(r"(?:因此|所以|综上|故|于是)[，,]?\s*(.+)")
    RE_CHOICE_WORD = re.compile(r"(?:选|答案是|answer\s*[:：]?\s*)\s*[\(（]?([A-E])[\)）]?\b",
                                re.IGNORECASE)
    RE_EQ_TAIL = re.compile(r"=\s*([^\n=]{1,80})\s*$")

    @staticmethod
    def _extract_boxed(text: str) -> Optional[str]:
        """提取最后一个 \\boxed{...}，正确处理嵌套花括号。"""
        key = r"\boxed"
        idx = text.rfind(key)
        if idx < 0:
            return None
        i = text.find("{", idx + len(key))
        if i < 0:
            return None
        depth, j = 1, i + 1
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        return text[i + 1:j - 1] if depth == 0 else None

    @staticmethod
    def _last_matches(pattern: re.Pattern, text: str) -> Optional[str]:
        hits = pattern.findall(text)
        if not hits:
            return None
        hit = hits[-1]
        return hit if isinstance(hit, str) else hit[0] if hit else None

    @classmethod
    def parse(cls, raw: str, analysis: Dict[str, Any]) -> Tuple[str, str]:
        """统一入口：核心抽取 + 关键词型答案规范化。"""
        ans, method = cls._parse_core(raw, analysis)
        if ans:
            ans = cls._finalize_kind(ans, raw or "", analysis)
        return ans, method

    @classmethod
    def _parse_core(cls, raw: str, analysis: Dict[str, Any]) -> Tuple[str, str]:
        text = (raw or "").strip()
        if not text:
            return "", "none"

        # 1) 首选显式契约："答案：X"
        hit = cls._last_matches(cls.RE_ANSWER_ZH, text)
        if hit and hit.strip():
            return cls._tidy(hit), "answer_zh"

        # 2) \boxed{...}
        boxed = cls._extract_boxed(text)
        if boxed and boxed.strip():
            return cls._tidy(boxed), "boxed"

        # 3) 英文 Answer 模式
        for pat, name in ((cls.RE_THE_ANSWER_IS, "the_answer_is"),
                          (cls.RE_ANSWER_EN, "answer_en")):
            hit = cls._last_matches(pat, text)
            if hit and hit.strip():
                return cls._tidy(hit), name

        # 4) 选择题字母（含 "选C" / 末尾单字母行）
        if analysis.get("is_choice"):
            hit = cls._last_matches(cls.RE_CHOICE_WORD, text)
            if hit:
                return hit.strip().upper(), "choice_word"
            tail_lines = [ln.strip() for ln in text.splitlines() if ln.strip()][-3:]
            for ln in reversed(tail_lines):
                m = re.fullmatch(r"[\(（]?([A-E])[\)）]?[.。]?", ln)
                if m:
                    return m.group(1).upper(), "choice_tail"

        # 5) 因此/所以 引导的结论
        hit = cls._last_matches(cls.RE_THEREFORE, text)
        if hit and hit.strip():
            return cls._tidy(hit), "therefore"

        # 6) 末行等号表达式
        hit = cls._last_matches(cls.RE_EQ_TAIL, text)
        if hit and hit.strip():
            return cls._tidy(hit), "eq_tail"

        # 7) 截断回捞：Thinking Process 被 max_tokens 截断时，抓最后一个数值/简单表达式
        recovered = cls._recover_from_truncation(text)
        if recovered:
            return cls._finalize_kind(recovered, text, analysis), "recover_truncation"

        # 8) 兜底：最后一个非空行（截断长度）
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            return cls._finalize_kind(cls._tidy(lines[-1]), text, analysis), "last_line"
        return "", "none"

    # 关键词型答案的规范化映射（按优先级排序，先匹配优先）
    KIND_CANON = {
        "convergence": [("不收敛", "发散"), ("条件收敛", "条件收敛"),
                        ("绝对收敛", "绝对收敛"), ("收敛", "收敛"), ("发散", "发散")],
        "hypothesis": [("不拒绝", "不拒绝 H0"), ("无法拒绝", "不拒绝 H0"),
                       ("拒绝", "拒绝 H0")],
    }

    @classmethod
    def _finalize_kind(cls, answer: str, raw: str, analysis: Dict[str, Any]) -> str:
        """对敛散性/假设检验等关键词型题目，把答案规范为判分期望的标准关键词。
        优先在已抽取的答案里找，其次在原文末尾找证据。"""
        rules = cls.KIND_CANON.get(analysis.get("answer_kind", ""))
        if not rules:
            return answer
        for kw, canon in rules:
            if kw in answer:
                return canon
        tail = (raw or "")[-1500:]
        for kw, canon in rules:
            if kw in tail:
                return canon
        return answer

    @staticmethod
    def _recover_from_truncation(text: str) -> Optional[str]:
        tail = text[-1200:]
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", tail)
        if nums:
            return nums[-1]
        return None

    @staticmethod
    def _tidy(ans: str) -> str:
        s = str(ans).strip()
        s = re.sub(r"\s+", " ", s)
        s = s.strip("*_` ")
        s = re.sub(r"^[。，,；;：:\s]+", "", s)
        s = re.sub(r"[。，,；;！!.\s]+$", "", s)
        if s.startswith("$") and s.endswith("$") and len(s) > 2:
            s = s[1:-1].strip()
        return s


# ==============================================================================
# 答案归一化与符号等价
# ==============================================================================

_FULL2HALF = str.maketrans({
    "（": "(", "）": ")", "，": ",", "：": ":", "；": ";",
    "＋": "+", "－": "-", "×": "*", "÷": "/", "。": ".",
    "％": "%", "−": "-", "–": "-",
})


def normalize_answer(ans: str) -> str:
    """保守归一化：只动格式，不动语义。"""
    s = str(ans or "").strip().translate(_FULL2HALF)
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("$", "").replace("π", "\\pi")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".,")
    return s


_RE_NUMBER = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_RE_FRAC = re.compile(r"^([+-]?\d+)\s*/\s*(\d+)$")
_RE_PERCENT = re.compile(r"^([+-]?(?:\d+(?:\.\d+)?))\s*%$")


def try_parse_number(ans: str) -> Optional[float]:
    s = normalize_answer(ans)
    m = _RE_PERCENT.match(s)
    if m:
        try:
            return float(m.group(1)) / 100.0
        except Exception:
            return None
    if _RE_NUMBER.match(s):
        try:
            return float(s)
        except Exception:
            return None
    m = _RE_FRAC.match(s)
    if m:
        try:
            return float(Fraction(int(m.group(1)), int(m.group(2))))
        except Exception:
            return None
    return None


# ==============================================================================
# 一题多解支持：多值答案拆分 / 集合级等价 / 完整性偏好
# ==============================================================================

def _strip_assignment(s: str) -> str:
    """'x = 2' -> '2'；仅当左侧是简短变量/表达式时取等号右侧。"""
    if "=" not in s:
        return s
    left, right = s.rsplit("=", 1)
    if len(left.strip()) <= 12 and right.strip():
        return right.strip()
    return s


def split_multi_answer(ans: str) -> Optional[List[str]]:
    """
    把多值答案拆成元素列表，处理：
      'x=2, x=3' / '{2,3}' / '2 或 3' / '2 和 3' / 'x=±2' / 'π/6, 5π/6'
    括号/区间内部不拆；元素不足 2 个返回 None（视为单值答案）。
    """
    s = str(ans or "").strip().translate(_FULL2HALF)
    s = s.replace("$", "").strip()
    # 剥离成对的外层集合括号：{2,3} / \{2,3\} -> 2,3（圆括号保留，可能是区间/坐标）
    if s.startswith("\\{") and s.endswith("\\}"):
        s = s[2:-2].strip()
    elif s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    if not s:
        return None
    # 顶层分隔：括号深度为 0 处的逗号/顿号/分号
    parts, buf, depth = [], [], 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch in ",,;;、" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    # 无括号片段再按 或/和/or 拆
    out: List[str] = []
    for p in parts:
        if not re.search(r"[(){}\[\]]", p):
            out.extend(re.split(r"或|和|\bor\b", p, flags=re.IGNORECASE))
        else:
            out.append(p)
    out = [_strip_assignment(p.strip()) for p in out if p.strip()]
    out = [p for p in out if p]
    # 展开 ±：'±2' -> ['2', '-2']
    final: List[str] = []
    for p in out:
        if "±" in p:
            final.append(p.replace("±", "").strip())
            final.append(p.replace("±", "-").strip())
        else:
            final.append(p)
    final = [p for p in final if p]
    return final if len(final) >= 2 else None


def _sets_equivalent(sa: List[str], sb: List[str], timeout: float) -> bool:
    """两个多值列表作为集合等价：每个元素都能在对方找到等价项（贪心匹配）。"""
    if len(sa) != len(sb):
        return False
    per = max(0.5, timeout / max(1, len(sa)))
    used = [False] * len(sb)
    for a in sa:
        hit = -1
        for j, b in enumerate(sb):
            if not used[j] and _single_equivalent(a, b, per):
                hit = j
                break
        if hit < 0:
            return False
        used[hit] = True
    return True


def _is_subset_answer(small: str, big: str, timeout: float = 4.0) -> bool:
    """small 的值集合是否为 big 的真子集（用于'求所有解'时的完整性偏好）。"""
    ss = split_multi_answer(small) or [normalize_answer(small)]
    sb = split_multi_answer(big) or [normalize_answer(big)]
    if len(ss) >= len(sb):
        return False
    per = max(0.5, timeout / max(1, len(ss)))
    used = [False] * len(sb)
    for a in ss:
        hit = -1
        for j, b in enumerate(sb):
            if not used[j] and _single_equivalent(a, b, per):
                hit = j
                break
        if hit < 0:
            return False
        used[hit] = True
    return True


_LATEX_FIXUPS = [
    (re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}"), r"(\1)/(\2)"),
    (re.compile(r"\\sqrt\[([^\]]+)\]\{([^{}]+)\}"), r"root(\2,\1)"),
    (re.compile(r"\\sqrt\{([^{}]+)\}"), r"sqrt(\1)"),
    (re.compile(r"\\pi\b"), r"pi"),
    (re.compile(r"\\infty"), r"oo"),
    (re.compile(r"\\cdot"), r"*"),
    (re.compile(r"\\times"), r"*"),
    (re.compile(r"\\div"), r"/"),
    (re.compile(r"\\log_?\{?([^{}]*)\}?"), r"log"),
    (re.compile(r"\\ln"), r"log"),
    (re.compile(r"\\sin|\\cos|\\tan|\\exp|\\abs"),
     lambda m: m.group(0)[1:]),
    (re.compile(r"e\^\{([^{}]+)\}"), r"exp(\1)"),
    (re.compile(r"\^\{([^{}]+)\}"), r"**(\1)"),
    (re.compile(r"\^"), r"**"),
    (re.compile(r"[{}]"), r""),
]

_TRANSFORMS = None
if _SYMPY_OK:
    _TRANSFORMS = standard_transformations + (
        convert_xor, implicit_multiplication_application)


def _to_sympy(ans: str):
    """尽力把答案字符串转成 sympy 表达式；失败返回 None。"""
    if not _SYMPY_OK:
        return None
    s = _strip_assignment(normalize_answer(ans))
    if not s or len(s) > 120:
        return None
    for pat, rep in _LATEX_FIXUPS:
        s = pat.sub(rep, s)
    s = s.replace("i", "I") if re.fullmatch(r"[0-9+\-*/() .iI]*", s) else s
    try:
        return parse_expr(s, transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        return None


def _single_equivalent(a: str, b: str, timeout: float = 4.0) -> bool:
    """单值等价判断：字符串归一化 -> 数值近似 -> sympy 符号化简。"""
    na, nb = _strip_assignment(normalize_answer(a)), _strip_assignment(normalize_answer(b))
    if not na or not nb:
        return False
    if na == nb or na.lower() == nb.lower():
        return True

    # 选项字母集合等价：'BD' == 'B, D' == 'D和B'（多选题）
    if re.fullmatch(r"[A-Ea-e,，、\s]+", str(a or "")) \
            and re.fullmatch(r"[A-Ea-e,，、\s]+", str(b or "")):
        la = sorted(set(re.findall(r"[A-E]", str(a).upper())))
        lb = sorted(set(re.findall(r"[A-E]", str(b).upper())))
        if la and lb:
            return la == lb

    fa, fb = try_parse_number(na), try_parse_number(nb)
    if fa is not None and fb is not None:
        return math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-9)

    if not _SYMPY_OK:
        return False

    def _check():
        ea, eb = _to_sympy(a), _to_sympy(b)
        if ea is None or eb is None:
            return False
        try:
            diff = sympy.simplify(ea - eb)
            return diff == 0
        except Exception:
            try:
                return bool(sympy.nsimplify(ea) == sympy.nsimplify(eb))
            except Exception:
                return False

    return bool(run_with_timeout(_check, timeout, default=False))


def answers_equivalent(a: str, b: str, timeout: float = 4.0) -> bool:
    """完整等价判断：先尝试单值三级等价，再尝试多值集合等价（一题多解）。"""
    if _single_equivalent(a, b, timeout):
        return True
    sa, sb = split_multi_answer(a), split_multi_answer(b)
    if sa and sb:
        return _sets_equivalent(sa, sb, timeout)
    return False


# ==============================================================================
# 自洽投票：Union-Find 聚类 + 权重表决
# ==============================================================================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


@dataclass
class VoteItem:
    answer: str
    source: str            # candidate_i / reflection_i / code_exec / judge
    weight: float = 1.0


@dataclass
class VoteResult:
    winner: str
    winner_score: float
    clusters: List[Dict[str, Any]]
    is_tie: bool


def cluster_and_vote(items: List[VoteItem], cfg: AgentConfig) -> VoteResult:
    """按符号等价聚类，输出得分最高的代表答案。"""
    valid = [it for it in items if it.answer and it.answer.strip()]
    if not valid:
        return VoteResult(winner="", winner_score=0.0, clusters=[], is_tie=False)

    n = len(valid)
    uf = UnionFind(n)
    eq_timeout = cfg.sympy_simplify_timeout if cfg.enable_equivalence else 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if eq_timeout > 0:
                if answers_equivalent(valid[i].answer, valid[j].answer, eq_timeout):
                    uf.union(i, j)
            elif normalize_answer(valid[i].answer).lower() \
                    == normalize_answer(valid[j].answer).lower():
                uf.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    clusters = []
    for members in groups.values():
        weight = sum(valid[i].weight for i in members)
        # 代表答案：组内出现次数最多；平票时优先值集合更完整的（一题多解），再取最短
        freq: Dict[str, int] = {}
        rep_pool: Dict[str, str] = {}
        for i in members:
            key = normalize_answer(valid[i].answer).lower()
            freq[key] = freq.get(key, 0) + 1
            rep_pool.setdefault(key, valid[i].answer)

        def _completeness(k: str) -> int:
            parts = split_multi_answer(rep_pool[k])
            return len(parts) if parts else 1

        best_key = max(freq, key=lambda k: (freq[k], _completeness(k), -len(k)))
        clusters.append({
            "representative": rep_pool[best_key],
            "weight": weight,
            "count": len(members),
            "sources": sorted({valid[i].source for i in members}),
        })

    clusters.sort(key=lambda c: (-c["weight"], -c["count"]))
    top = clusters[0]
    is_tie = len(clusters) > 1 and abs(clusters[1]["weight"] - top["weight"]) < 1e-9
    total = sum(c["weight"] for c in clusters)
    return VoteResult(
        winner=top["representative"],
        winner_score=top["weight"] / total if total else 0.0,
        clusters=clusters,
        is_tie=is_tie,
    )


# ==============================================================================
# SymPy 代码验算：LLM 生成脚本 -> 受限沙箱执行
# ==============================================================================

class SymPyCodeVerifier:
    """
    让模型为题目写一段 sympy 计算脚本，沙箱执行后把 print 结果作为独立票源。
    安全策略：AST 黑名单 + 受限 builtins + 超时执行，不接触文件/网络/系统。
    """

    FORBIDDEN = re.compile(
        r"\b(os|sys|subprocess|open|exec|eval|input|__import__|import\s+os|"
        r"socket|requests|urllib|shutil|pathlib|pickle|marshal|"
        r"globals|locals|getattr|setattr|delattr|compile|breakpoint|exit|quit)\b")

    RE_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

    ALLOWED_IMPORTS = {"math", "sympy", "fractions", "itertools",
                       "functools", "cmath", "numbers", "collections"}
    STRIPPED_BUILTINS = {"open", "exec", "eval", "input", "compile",
                         "breakpoint", "exit", "quit", "help", "memoryview"}

    @classmethod
    def extract_code(cls, text: str) -> Optional[str]:
        if not text:
            return None
        m = cls.RE_CODE_BLOCK.findall(text)
        code = m[-1] if m else (text if "print" in text else None)
        if not code:
            return None
        return cls._repair_indent(code.strip())

    @staticmethod
    def _repair_indent(code: str) -> str:
        """LLM 常输出整体多缩进/首行未缩进的代码，逐一尝试修复到可编译。"""
        import textwrap
        candidates = [code, textwrap.dedent(code)]
        lines = code.split("\n")
        first = lines[0] if lines else ""
        if first and not first.startswith((" ", "\t")) \
                and not first.rstrip().endswith((":", "\\")):
            rest = [ln for ln in lines[1:] if ln.strip()]
            if rest:
                ind = min(len(ln) - len(ln.lstrip()) for ln in rest)
                if ind > 0:
                    fixed = [first] + [ln[ind:] if ln.strip() else ln
                                       for ln in lines[1:]]
                    candidates.append("\n".join(fixed))
        for cand in candidates:
            try:
                compile(cand, "<sandbox>", "exec")
                return cand
            except Exception:
                continue
        return code

    @classmethod
    def _is_safe(cls, code: str) -> bool:
        if cls.FORBIDDEN.search(code):
            return False
        try:
            import ast as _ast
            tree = _ast.parse(code)
        except Exception:
            return False
        import ast as _ast
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                names = [a.name.split(".")[0] for a in node.names]
                root = (getattr(node, "module", "") or "").split(".")[0]
                if root and root not in cls.ALLOWED_IMPORTS:
                    return False
                if any(nm not in cls.ALLOWED_IMPORTS for nm in names) and not root:
                    return False
        return True

    # ------------------------------------------------------------------ 进程级沙箱
    @staticmethod
    def _child_exec(code: str, q) -> None:
        """子进程入口：受限 builtins + 导入守卫，print 输出经队列回传。"""
        try:
            import builtins as _b
            bd = {k: getattr(_b, k) for k in dir(_b)}
            for name in SymPyCodeVerifier.STRIPPED_BUILTINS:
                bd.pop(name, None)
            real_import = _b.__import__

            def guarded_import(name, *a, **k):
                root = str(name).split(".")[0]
                if root not in SymPyCodeVerifier.ALLOWED_IMPORTS:
                    raise ImportError(f"module not allowed: {root}")
                return real_import(name, *a, **k)

            lines: List[str] = []
            bd["__import__"] = guarded_import
            bd["print"] = lambda *a, **k: lines.append(" ".join(str(x) for x in a))
            g = {"__builtins__": bd, "math": math, "sp": sympy, "sympy": sympy}
            try:
                exec(compile(code, "<sandbox>", "exec"), g)  # noqa: S102 - 已静态黑名单+进程隔离
                q.put(("ok", "\n".join(lines).strip()))
            except Exception as exc:  # noqa: BLE001
                q.put(("error", f"{type(exc).__name__}: {exc}"[:200]))
        except Exception as exc:  # noqa: BLE001
            try:
                q.put(("error", f"sandbox: {exc}"[:200]))
            except Exception:
                pass

    @classmethod
    def execute(cls, code: str, timeout: float) -> Tuple[str, str]:
        """返回 (stdout_text, status)。status ∈ ok / unsafe / error / timeout / no_output"""
        if not _SYMPY_OK:
            return "", "error"
        if not cls._is_safe(code):
            return "", "unsafe"

        status, out = cls._exec_via_process(code, timeout)
        if status is None:  # multiprocessing 不可用 -> 线程退化方案
            status, out = cls._exec_via_thread(code, timeout)

        out = (out or "").strip()
        if status == "ok":
            if not out:
                return "", "no_output"
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            return (lines[-1] if lines else ""), "ok"
        return "", status

    @classmethod
    def _exec_via_process(cls, code: str, timeout: float):
        """(status|None, output)。status 为 None 表示 multiprocessing 不可用。"""
        try:
            import multiprocessing as mp
            q = mp.Queue()
            p = mp.Process(target=cls._child_exec, args=(code, q), daemon=True)
            p.start()
            p.join(max(timeout, 0.1))
            if p.is_alive():
                p.terminate()
                p.join(1.0)
                return "timeout", ""
            try:
                return q.get_nowait()
            except Exception:
                return "error", ""
        except Exception:
            return None, ""

    @classmethod
    def _exec_via_thread(cls, code: str, timeout: float):
        """退化方案：线程 + 自定义 print（不做全局 stdout 重定向）。"""
        box = {"done": False, "status": "error", "out": ""}

        def _target():
            lines: List[str] = []
            g = {"__builtins__": __builtins__, "math": math,
                 "sp": sympy, "sympy": sympy,
                 "print": lambda *a, **k: lines.append(" ".join(str(x) for x in a))}
            try:
                exec(compile(code, "<sandbox>", "exec"), g)  # noqa: S102
                box.update(done=True, status="ok", out="\n".join(lines))
            except Exception as exc:  # noqa: BLE001
                box.update(done=True, status="error", out=str(exc)[:200])

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(max(timeout, 0.1))
        if not box["done"]:
            return "timeout", ""
        return box["status"], box["out"]


# ==============================================================================
# 答案契约：题型 -> 合格答案的硬性要求
# ==============================================================================

class AnswerContract:
    """契约不满足 -> 触发反思纠错或降级选择。"""

    @staticmethod
    def check(answer: str, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        ans = (answer or "").strip()
        if not ans:
            return False, "empty"
        kind = analysis.get("answer_kind", "value")

        if kind == "letter":
            if analysis.get("multi_select"):
                return (bool(re.fullmatch(r"[A-E]{1,5}", ans)), "expect_letters")
            return (bool(re.fullmatch(r"[A-E]", ans)), "expect_letter")
        if kind == "convergence":
            return (("收敛" in ans) or ("发散" in ans), "expect_conv_div")
        if kind == "hypothesis":
            return ("拒绝" in ans or "不拒绝" in ans or "reject" in ans.lower(),
                    "expect_reject")
        if kind == "interval":
            return (bool(re.search(r"[\[(].+,.+[\])]", ans)), "expect_interval")
        if kind == "proof":
            return (len(ans) >= 2, "expect_nonempty_conclusion")
        if kind == "judgment":
            return (bool(re.search(r"是|否|正确|错误|成立|不成立|yes|no", ans,
                                   re.IGNORECASE)), "expect_judgment")
        # value：排除明显的非答案文本
        if re.search(r"thinking|deliberation|analysis", ans, re.IGNORECASE) \
                and len(ans) > 60:
            return False, "looks_like_thinking"
        return True, "ok"


# ==============================================================================
# 最终格式化
# ==============================================================================

class AnswerFormatter:
    @staticmethod
    def finalize(answer: str, analysis: Dict[str, Any], max_chars: int) -> str:
        ans = (answer or "").strip()
        kind = analysis.get("answer_kind", "value")
        if kind == "letter":
            letters = sorted(set(re.findall(r"[A-E]", ans.upper())))
            if letters:
                return "".join(letters) if analysis.get("multi_select") else letters[0]
        # 去除包装词
        ans = re.sub(r"^(?:最终答案|答案|answer)\s*(?:为|是|[:：])\s*", "",
                     ans, flags=re.IGNORECASE).strip()
        ans = ans.rstrip("。.;；，,")
        if len(ans) > max_chars:  # 超长护栏：优先截取首个表达式
            head = ans[:max_chars]
            ans = head.rsplit("\n", 1)[-1] or head
        return ans or "无法确定"


# ==============================================================================
# 主智能体
# ==============================================================================

class ReasoningAgent:
    """
    平台入口类。构造与 solve 签名为比赛强制规范，请勿修改。

    解题流水线：
        analyze -> generate(N 候选) -> parse/normalize -> [sympy 代码验算]
        -> cluster vote -> [契约检查 / 平票裁判 / 反思重试] -> finalize
    任何阶段异常都会兜底，final_response 保证非空。
    """

    def __init__(self, client, *args, **kwargs):
        self.client = client
        cfg = kwargs.get("config") or (args[0] if args else None)
        self.config = cfg if isinstance(cfg, AgentConfig) else AgentConfig()
        self.llm = LLMCaller(client, self.config)

    # ------------------------------------------------------------------ 入口
    def solve(self, problem: str, metadata: dict) -> dict:
        metadata = metadata or {}
        idx = metadata.get("idx", 0)
        started = time.time()
        trace: List[Dict[str, Any]] = []

        def tlog(step: str, content: Any):
            trace.append({"step": step, "content": json_safe(content)})

        try:
            result = self._solve_inner(problem, idx, started, trace, tlog)
        except Exception as exc:  # noqa: BLE001 - 顶层兜底，绝不让异常逃逸出 solve
            tlog("fatal_fallback", f"{type(exc).__name__}: {exc}")
            result = {"final_response": "无法确定", "trace": trace}

        result["final_response"] = str(result.get("final_response") or "无法确定")
        if not result["final_response"].strip():
            result["final_response"] = "无法确定"
        result["trace"] = json_safe(result.get("trace") or trace)
        return result

    # ------------------------------------------------------------------ 主流程
    def _solve_inner(self, problem: str, idx: int, started: float,
                     trace: List[Dict[str, Any]], tlog) -> Dict[str, Any]:
        cfg = self.config

        def time_left() -> float:
            return cfg.time_budget_sec - (time.time() - started)

        # ---- 1. 题目分析 ----
        analysis = ProblemAnalyzer.analyze(problem)
        tlog("analyze", analysis)

        # ---- 2. 候选生成（并发多温度采样）----
        n_cand = max(2, min(cfg.num_candidates, cfg.max_candidates_hard_cap))
        raw_candidates = self._generate_candidates(problem, analysis, n_cand, tlog)

        # ---- 3. 答案抽取与归一化 ----
        votes: List[VoteItem] = []
        parsed_info = []
        for i, raw in enumerate(raw_candidates):
            ans, method = AnswerParser.parse(raw, analysis)
            parsed_info.append({"idx": i, "answer": clip(ans, 80), "method": method})
            if ans.strip():
                votes.append(VoteItem(answer=ans, source=f"candidate_{i}"))
        tlog("parsed_candidates", parsed_info)

        # ---- 4. SymPy 代码验算（独立票源，仅对计算型题目启用）----
        # 选择/判断/敛散/假设检验/证明题的答案不是算出来的，验算票源
        # 只会浪费 1 次 LLM 调用并可能引入噪声票 -> 直接跳过。
        _CODE_KINDS = {"value", "interval"}
        if cfg.enable_code_verification and time_left() > 60 \
                and analysis.get("answer_kind") in _CODE_KINDS:
            self._code_verification(problem, votes, tlog)

        # ---- 5. 自洽聚类投票 ----
        vote = cluster_and_vote(votes, cfg)
        tlog("vote", {
            "winner": clip(vote.winner, 80),
            "winner_score": round(vote.winner_score, 3),
            "is_tie": vote.is_tie,
            "clusters": [
                {"answer": clip(c["representative"], 60),
                 "weight": c["weight"], "count": c["count"]}
                for c in vote.clusters[:5]
            ],
        })

        # ---- 6. 平票裁判（仅必要时，1 次调用）----
        if vote.is_tie and cfg.enable_judge_on_tie and len(vote.clusters) >= 2 \
                and time_left() > 45:
            vote = self._judge_tie(problem, vote, tlog)

        # ---- 7. 契约检查 + 反思纠错 ----
        need_reflect, reason = self._should_reflect(vote, analysis, votes)
        round_no = 0
        while need_reflect and cfg.enable_reflection \
                and round_no < cfg.max_reflect_rounds and time_left() > 90:
            round_no += 1
            tlog("reflect_trigger", {"round": round_no, "reason": reason})
            new_votes = self._reflect_round(problem, analysis, vote, round_no, tlog)
            votes.extend(new_votes)
            vote = cluster_and_vote(votes, cfg)
            tlog("vote_after_reflect", {
                "round": round_no,
                "winner": clip(vote.winner, 80),
                "winner_score": round(vote.winner_score, 3),
            })
            need_reflect, reason = self._should_reflect(vote, analysis, votes)

        # ---- 8. 一题多解完整性偏好：题目要求"所有解"时，真超集簇优先 ----
        if analysis.get("wants_all") and len(vote.clusters) > 1:
            vote = self._prefer_complete_answer(vote, tlog)

        # ---- 9. 最终格式化 ----
        winner = vote.winner
        ok, contract_msg = AnswerContract.check(winner, analysis)
        if not ok:
            # 契约仍不满足：按簇权重从高到低找满足契约的答案降级使用
            replacement = self._best_contract_passing(vote, votes, analysis)
            if replacement:
                tlog("contract_fallback",
                     {"from": clip(winner, 60), "to": clip(replacement, 60),
                      "reason": contract_msg})
                winner = replacement
        final = AnswerFormatter.finalize(winner, analysis, cfg.max_answer_chars)

        tlog("finalize", {
            "final_response": clip(final, 120),
            "contract_ok": ok,
            "llm_calls": self.llm.calls,
            "llm_errors": self.llm.errors,
            "elapsed_sec": round(time.time() - started, 1),
        })
        return {"final_response": final, "trace": trace}

    # ------------------------------------------------------------------ 候选生成
    def _generate_candidates(self, problem: str, analysis: Dict[str, Any],
                             n: int, tlog) -> List[str]:
        cfg = self.config
        messages = PromptEngineer.solver_messages(problem, analysis)
        temps = [cfg.candidate_temperatures[i % len(cfg.candidate_temperatures)]
                 for i in range(n)]
        results: List[Optional[str]] = [None] * n

        def _one(i: int) -> Tuple[int, str, str]:
            try:
                return i, self.llm.chat(messages, temperature=temps[i],
                                        max_tokens=cfg.max_tokens), ""
            except Exception as exc:  # noqa: BLE001
                return i, "", f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=max(1, min(cfg.gen_workers, n))) as pool:
            futures = [pool.submit(_one, i) for i in range(n)]
            for fut in as_completed(futures):
                i, text, err = fut.result()
                results[i] = text
                if err:
                    tlog(f"candidate_{i}_error", clip(err, 200))

        cands = [r or "" for r in results]
        tlog("candidates_raw", [
            {"idx": i, "chars": len(c), "head": clip(c, cfg.trace_candidate_chars)}
            for i, c in enumerate(cands)
        ])
        return cands

    # ------------------------------------------------------------------ 代码验算
    def _code_verification(self, problem: str, votes: List[VoteItem],
                           tlog) -> None:
        cfg = self.config
        try:
            text = self.llm.chat(PromptEngineer.code_messages(problem),
                                 temperature=0.0, max_tokens=cfg.code_max_tokens)
        except Exception as exc:  # noqa: BLE001
            tlog("code_verification", {"status": "llm_error", "err": clip(exc, 150)})
            return
        code = SymPyCodeVerifier.extract_code(text)
        if not code:
            tlog("code_verification", {"status": "no_code"})
            return
        out, status = SymPyCodeVerifier.execute(code, cfg.code_exec_timeout)
        tlog("code_verification", {
            "status": status,
            "code": clip(code, cfg.trace_candidate_chars),
            "output": clip(out, 120),
        })
        if status == "ok" and out.strip():
            ans_out = out.strip()
            # sympy solve 等输出的列表 [2, 3] -> 2, 3，便于与文本答案集合等价
            m_list = re.fullmatch(r"\[(.*)\]", ans_out, flags=re.DOTALL)
            if m_list and not re.search(r"Matrix|Interval", ans_out):
                ans_out = m_list.group(1).strip()
            votes.append(VoteItem(answer=ans_out, source="code_exec",
                                  weight=float(cfg.code_vote_weight)))

    # ------------------------------------------------------------------ 平票裁判
    def _judge_tie(self, problem: str, vote: VoteResult, tlog) -> VoteResult:
        cfg = self.config
        options = [c["representative"] for c in vote.clusters[:4]]
        try:
            text = self.llm.chat(PromptEngineer.judge_messages(problem, options),
                                 temperature=cfg.judge_temperature, max_tokens=2048)
        except Exception as exc:  # noqa: BLE001
            tlog("judge", {"status": "llm_error", "err": clip(exc, 150)})
            return vote
        m = re.findall(r"VERDICT\s*[:：]\s*([A-Z])", text, flags=re.IGNORECASE)
        picked = None
        if m:
            k = ord(m[-1].upper()) - 65
            if 0 <= k < len(options):
                picked = options[k]
        tlog("judge", {"picked": clip(picked, 60), "raw": clip(text, 200)})
        if picked:
            for c in vote.clusters:
                if c["representative"] == picked or \
                        answers_equivalent(c["representative"], picked,
                                           cfg.sympy_simplify_timeout):
                    c["weight"] += 1.5
                    vote.clusters.sort(key=lambda x: (-x["weight"], -x["count"]))
                    vote.winner = vote.clusters[0]["representative"]
                    break
        return vote

    # ------------------------------------------------------------------ 反思
    def _should_reflect(self, vote: VoteResult, analysis: Dict[str, Any],
                        votes: List[VoteItem]) -> Tuple[bool, str]:
        if not vote.winner.strip():
            return True, "no_answer"
        # 有效候选不足一半 -> 反思
        n_cand = max(1, min(self.config.num_candidates,
                            self.config.max_candidates_hard_cap))
        n_parsed = len([v for v in votes if v.source.startswith("candidate")])
        if n_parsed < max(2, n_cand // 2):
            return True, "parse_failures"
        ok, msg = AnswerContract.check(vote.winner, analysis)
        if not ok:
            return True, f"contract_fail:{msg}"
        if vote.is_tie:
            return True, "tie_unresolved"
        # 多数优势不足 50% 且存在多个簇 -> 值得反思一次
        if vote.winner_score < 0.5 and len(vote.clusters) > 1:
            return True, "weak_majority"
        return False, ""

    def _reflect_round(self, problem: str, analysis: Dict[str, Any],
                       vote: VoteResult, round_no: int, tlog) -> List[VoteItem]:
        cfg = self.config
        distinct = [c["representative"] for c in vote.clusters[:5]]
        messages = PromptEngineer.reflection_messages(problem, analysis, distinct)
        out: List[VoteItem] = []
        for k in range(2):  # 每轮补 2 个反思候选
            try:
                text = self.llm.chat(messages, temperature=cfg.reflect_temperature,
                                     max_tokens=cfg.max_tokens)
                ans, method = AnswerParser.parse(text, analysis)
                tlog(f"reflect_{round_no}_{k}", {
                    "answer": clip(ans, 80), "method": method,
                    "head": clip(text, cfg.trace_candidate_chars // 2),
                })
                if ans.strip():
                    out.append(VoteItem(answer=ans, source=f"reflection_{round_no}",
                                        weight=1.0))
            except Exception as exc:  # noqa: BLE001
                tlog(f"reflect_{round_no}_{k}_error", clip(exc, 150))
        return out

    # ------------------------------------------------------------------ 完整性偏好
    def _prefer_complete_answer(self, vote: VoteResult, tlog) -> VoteResult:
        """
        题目要求'所有解/解集'时：若某簇的值集合是 winner 的真超集，
        且其权重不低于 winner 的一半，则改用更完整的答案。
        例如 winner='x=2'，存在簇 '{2, -2}' -> 选后者。
        """
        winner = vote.winner
        win_weight = vote.clusters[0]["weight"] if vote.clusters else 0.0
        for c in vote.clusters[1:]:
            if c["weight"] + 1e-9 < win_weight * 0.5:
                break  # 簇已按权重降序，后面权重更低
            if _is_subset_answer(winner, c["representative"],
                                 self.config.sympy_simplify_timeout):
                tlog("completeness_prefer", {
                    "from": clip(winner, 60), "to": clip(c["representative"], 60)})
                vote.winner = c["representative"]
                return vote
        return vote

    # ------------------------------------------------------------------ 契约降级
    @staticmethod
    def _best_contract_passing(vote: VoteResult, votes: List[VoteItem],
                               analysis: Dict[str, Any]) -> Optional[str]:
        # 先按簇权重从高到低找（簇已按权重排序）
        for c in vote.clusters:
            ok, _ = AnswerContract.check(c["representative"], analysis)
            if ok:
                return c["representative"]
        # 再看零散票
        seen = {normalize_answer(c["representative"]).lower() for c in vote.clusters}
        for v in votes:
            key = normalize_answer(v.answer).lower()
            if key in seen:
                continue
            seen.add(key)
            ok, _ = AnswerContract.check(v.answer, analysis)
            if ok:
                return v.answer
        return None


# ==============================================================================
# 本地自测入口（python user_agent.py 不会触发任何网络调用）
# ==============================================================================

if __name__ == "__main__":
    print("user_agent.py loaded OK. ReasoningAgent is ready for the platform runner.")
