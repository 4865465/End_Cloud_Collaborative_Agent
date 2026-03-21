"""
Evaluation Module using LLM as a Judge
整合了 0-10 质量评分与 0/1 语义一致性检测
"""
import re
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from llm_client import LLMClient  # 假设已根据你的修改建议实现
from config import EVALUATION_LLM_API_KEY, EVALUATION_LLM_API_BASE, EVALUATION_LLM_MODEL

logger = logging.getLogger(__name__)

class Evaluator:
    """
    统一评估器：支持对话质量评分 (0-10) 和 答案一致性判断 (0/1)。
    """
    
    def __init__(self):
        try:
            self.client = LLMClient(
                api_key=EVALUATION_LLM_API_KEY,
                base_url=EVALUATION_LLM_API_BASE,
                model_name=EVALUATION_LLM_MODEL
            )
            logger.info(f"成功初始化评估器，模型: {EVALUATION_LLM_MODEL}")
        except Exception as e:
            self.client = None
            logger.error(f"评估器初始化失败: {e}")

    def _remove_think_content(self, text: str) -> str:
        """清理推理模型（如 DeepSeek R1）的思维链内容"""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _extract_score(self, text: str) -> float:
        """从模型回复中安全地提取最后一个数字"""
        content = self._remove_think_content(text)
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', content)
        if not numbers:
            return 0.0
        return float(numbers[-1])

    def calculate_llm_score(
        self,
        question: str,
        candidate: str,
        reference: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 3
    ) -> float:
        """
        核心评分逻辑。
        - 如果提供 reference: 执行 0/1 一致性判断。
        - 如果不提供 reference: 执行 0-10 质量评分。
        """
        # print("question: ",question)
        # print("candidate: ",candidate)
        # print("history: ",history)
        if not self.client:
            return 0.0

        # 模式选择与 Prompt 构建
        if reference:
            # 模式 A: 语义一致性判断 (来自 MSEvaluator)
            prompt = f"""
Please determine whether the response is semantically consistent with the standard answer and whether it correctly answers the question.
Output your judgment using a number only:
1 = consistent
0 = not consistent
Do not include any other information or explanation. Output only the number.
Question: {question}
Standard Answer: {reference}
Response: {candidate}
Judgment:
"""
        else:
            # 模式 B: 质量评分 (来自 Evaluator)
            history_text = ""
            if history:
                lines = [f"{msg.get('role', '').capitalize()}: {msg.get('content', '')}" for msg in history]
                history_text = "\n".join(lines)
            
            prompt = self._get_quality_prompt(question, candidate, history_text)

        messages = [
            {"role": "system", "content": "You are an expert evaluator with deep knowledge in assessing AI-generated responses."},
            {"role": "user", "content": prompt}
        ]

        # 指数退避重试逻辑
        for attempt in range(max_retries):
            try:
                # 调用 LLMClient
                response = self.client.generate(messages, temperature=0.0)
                content = response.get("content", "")
                print(f'评估模型输出：{content}')
                score = self._extract_score(content)
                
                # 约束评分范围
                if reference:
                    return 1.0 if score >= 0.5 else 0.0
                return max(0.0, min(10.0, score))
                
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"评估出错，{wait}s 后重试 ({attempt+1}/{max_retries}): {e}")
                time.sleep(wait)
        
        return 0.0

    def _get_quality_prompt(self, question: str, candidate: str, history_text: str) -> str:
        """改进的评估提示词：评估当前轮次的回答质量，历史上下文仅用于理解背景"""
        if history_text:
            return f"""You will be given a conversation history and the current turn (user question and system answer).

IMPORTANT: You are evaluating ONLY the quality of the CURRENT TURN's system answer. The conversation history is provided ONLY to help you understand the context of the current question. Do NOT evaluate the entire conversation - focus solely on how well the current system answer addresses the current user question.

Your task is to provide a score (0-10) rating how well the CURRENT system answer addresses the CURRENT user question, considering the context provided by the conversation history.

Give your answer on a scale of 0 to 10, where:
- 0 means the current answer is not helpful at all, irrelevant, or severely misunderstandings the current question (even with context)
- 10 means the current answer is excellent, highly relevant, correct, detailed, and fully addresses the current user question

Here is the detailed scale:

0-2: Very poor — the current answer is largely irrelevant to the current question, off-topic, or severely incomplete; may contain serious misunderstandings; fails to address the current question even with context from history.
3-4: Poor — the current answer touches on the current question but misses many key aspects; has notable issues in correctness or clarity in addressing the current question.
5-6: Acceptable — the current answer is mostly on-topic and somewhat helpful for the current question, but with clear gaps in completeness, accuracy, or usefulness.
7-8: Good — the current answer is generally helpful, reasonably accurate and clear in addressing the current question; only minor issues or missing details.
9-10: Excellent — the current answer is highly relevant, correct, detailed, and fully addresses the current user question.

Note: Use the conversation history only to understand what the current question refers to (e.g., if it says "the previous topic" or "what you mentioned earlier"). Your evaluation should focus on the quality of the current answer to the current question.

Conversation History (for context only):
{history_text}

Current Turn (evaluate this):
User Question: {question}
System Answer: {candidate}

If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Output ONLY a number between 0 and 10 (integer). Do not include any other text, explanation, or formatting. Only output the score number.!!!"""
        else:
            return f"""You will be given a user question and system answer.

Your task is to provide a score (0-10) rating how well the system answer addresses the user question.

Give your answer on a scale of 0 to 10, where 0 means that the system answer is not helpful at all, and 10 means that the system answer completely and helpfully addresses the user question in an excellent way.

Here is the scale you should use:

0-2: Very poor — largely irrelevant, off-topic, or severely incomplete; may contain serious misunderstandings.
3-4: Poor — touches on the question but misses many key aspects, or has notable issues in correctness or clarity.
5-6: Acceptable — mostly on-topic and somewhat helpful, but with clear gaps in completeness, accuracy, or usefulness.
7-8: Good — generally helpful, reasonably accurate and clear; only minor issues or missing details.
9-10: Excellent — highly relevant, correct, detailed, and fully addresses the user's concerns.

User Question: {question}
System Answer: {candidate}

If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Output ONLY a number between 0 and 10 (integer). Do not include any other text, explanation, or formatting. Only output the score number.!!!"""

    def evaluate(
        self,
        question: str,                      # 将 question 移到第一位
        candidate: str,                     # 候选答案（模型输出）
        history: Optional[List[Dict[str, str]]] = None, # 历史记录移到第三位
        reference: Optional[str] = None     # 参考答案（可选）
    ) -> Dict[str, Any]:
        """对外保持一致的单条评估接口"""
        if not question:
            print('没有问题')
            return {"llm_score": 0.0}
        
        score = self.calculate_llm_score(question, candidate, reference, history)
        return {"llm_score": score}

    def evaluate_batch(
        self,
        references: List[Optional[str]],
        candidates: List[str],
        questions: List[str],
        histories: Optional[List[List[Dict[str, str]]]] = None
    ) -> Dict[str, Any]:
        """批量评估接口"""
        scores = []
        for i in range(len(questions)):
            hist = histories[i] if histories else None
            res = self.evaluate(references[i], candidates[i], questions[i], hist)
            scores.append(res["llm_score"])
        
        return {
            "average": {"llm_score": np.mean(scores)},
            "individual_scores": {"llm_score": scores}
        }