"""
Self-Reflection 反思自检模块 — 方案 A：纯文本语义校验

校验方式：
  - 仅调用大模型 + Prompt，做语义层面的「答案是否被参考资料支撑」判断
  - 不调用 Embedding API
  - 不计算向量距离 / 余弦相似度

工作流：
  初稿答案 → reflect_answer() → 通过则放行 / 不通过则采用修正答案
"""

import json
import re
from dataclasses import dataclass

from llm import chat
from rag.config import NOT_FOUND_REPLY


@dataclass
class ReflectionResult:
    """一次反思自检的结果。"""

    passed: bool
    reason: str
    draft_answer: str
    final_answer: str
    corrected: bool


# 质检 Prompt：纯文本输入输出，模型自行做语义比对
REFLECTION_PROMPT = """你是答案质检员。请只做「语义对照」，判断「待检查答案」是否被「参考资料」充分支撑。

【校验要求】
1. 逐条核对：待检查答案里的每个事实，能否在参考资料中找到语义对应（允许同义表述，不允许凭空添加）。
2. 幻觉判定：若答案包含参考资料完全没有的信息，判定为不通过。
3. 答非所问：若未回应用户问题，判定为不通过。
4. 本步骤禁止编造：修正答案只能使用参考资料中的信息。

【参考资料】
{context}

【用户问题】
{question}

【待检查答案】
{draft}

请只输出 JSON，不要 markdown 代码块，不要其他说明：
{{"pass": true或false, "reason": "一句话说明", "corrected_answer": "不通过时给出基于参考资料的修正回答；通过时留空字符串"}}"""


def _parse_reflection_output(raw: str) -> dict:
    """从大模型返回的纯文本中解析 JSON。"""
    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("反思自检返回格式异常，无法解析 JSON")
    return json.loads(match.group())


def reflect_answer(question: str, context: str, draft_answer: str) -> ReflectionResult:
    """
    方案 A：纯 Prompt 语义自检（无 Embedding、无向量距离）。

    Args:
        question: 用户原始问题
        context: 检索得到的参考资料纯文本（已拼接）
        draft_answer: RAG 生成的初稿答案

    Returns:
        ReflectionResult，含是否通过、原因、最终答案
    """
    empty_context = not context.strip() or context == "（未检索到相关内容）"
    if empty_context:
        return ReflectionResult(
            passed=True,
            reason="无参考资料，跳过语义对照",
            draft_answer=draft_answer,
            final_answer=draft_answer,
            corrected=False,
        )

    # 唯一的外部调用：llm.chat，走纯文本 Prompt 逻辑
    raw = chat(
        [
            {
                "role": "user",
                "content": REFLECTION_PROMPT.format(
                    context=context,
                    question=question,
                    draft=draft_answer,
                ),
            }
        ]
    )

    parsed = _parse_reflection_output(raw)
    passed = bool(parsed.get("pass", False))
    reason = str(parsed.get("reason", "")).strip() or "未提供原因"
    corrected = str(parsed.get("corrected_answer", "")).strip()

    if passed:
        return ReflectionResult(
            passed=True,
            reason=reason,
            draft_answer=draft_answer,
            final_answer=draft_answer,
            corrected=False,
        )

    final = corrected if corrected else NOT_FOUND_REPLY
    return ReflectionResult(
        passed=False,
        reason=reason,
        draft_answer=draft_answer,
        final_answer=final,
        corrected=True,
    )
