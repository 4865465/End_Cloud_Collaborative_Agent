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
        references: Optional[Dict[str, str]] = None,
        checklist: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 3
    ) -> float:
        """
        核心评分逻辑。
        使用 WildBench 的 checklist 和 references 进行评估。
        """
        if not self.client:
            return 0.0

        history_text = ""
        if history:
            lines = [f"{msg.get('role', '').capitalize()}: {msg.get('content', '')}" for msg in history]
            history_text = "\n".join(lines)
        
        # Format references
        ref_text = "No reference answers available."
        if references:
            ref_list = []
            for model, resp in references.items():
                ref_list.append(f"--- Reference Answer (from {model}) ---\n{resp}")
            ref_text = "\n\n".join(ref_list)
            
        # Format checklist
        checklist_text = "No specific checklist provided."
        if checklist:
            checklist_text = "\n".join([f"- {item}" for item in checklist])

        prompt = f"""You are an expert evaluator assessing the quality of an AI's response to a user question.
You will be provided with:
1. Conversation History (if any)
2. User Question
3. Reference Answers (provided by other strong models)
4. A Checklist of criteria to consider
5. The Candidate Response to be evaluated

Your goal is to provide a score between 0 and 10 based on how well the candidate response addresses the user question, following the provided checklist and comparing it with the reference answers.

### Context and Input
[Conversation History]
{history_text if history_text else "None"}

[User Question]
{question}

[Reference Answers]
{ref_text}

[Checklist]
{checklist_text}

### Candidate Response
{candidate}

### Evaluation Criteria
- 0: The response is completely irrelevant, incorrect, or harmful.
- 1-2: Very poor; fails to address the main request or contains major errors.
- 3-4: Poor; addresses some parts but misses key requirements or has notable inaccuracies.
- 5-6: Fair; covers the basics but lacks depth, clarity, or has minor errors.
- 7-8: Good; helpful, accurate, and follows most requirements.
- 9-10: Excellent; outstanding quality, fully addresses the question and meets all checklist items effectively.

Output ONLY a single integer between 0 and 10 representing the score. Do not include any explanations or other text.
Score:"""

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
                print(f'评估模型输出：{content[:100]}...')
                score = self._extract_score(content)
                
                return max(0.0, min(10.0, score))
                
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"评估出错，{wait}s 后重试 ({attempt+1}/{max_retries}): {e}")
                time.sleep(wait)
        
        return 0.0

    def evaluate(
        self,
        question: str,
        candidate: str,
        references: Optional[Dict[str, str]] = None,
        checklist: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """对外保持一致的单条评估接口"""
        if not question:
            print('没有问题')
            return {"llm_score": 0.0}
        # print(f'问题：{question}')
        # print(f'候选答案：{candidate}')
        # print(f'参考答案：{references}')
        # print(f'检查项：{checklist}')
        # print(f'对话历史：{history}')
        score = self.calculate_llm_score(question, candidate, references, checklist, history)
        return {"llm_score": score}

    def evaluate_batch(
        self,
        questions: List[str],
        candidates: List[str],
        references: List[Optional[Dict[str, str]]],
        checklists: Optional[List[List[str]]] = None,
        histories: Optional[List[List[Dict[str, str]]]] = None
    ) -> Dict[str, Any]:
        """批量评估接口"""
        scores = []
        for i in range(len(questions)):
            hist = histories[i] if histories else None
            checklist = checklists[i] if checklists else None
            res = self.evaluate(
                questions[i], 
                candidates[i], 
                references=references[i], 
                checklist=checklist, 
                history=hist
            )
            scores.append(res["llm_score"])
        
        return {
            "average": {"llm_score": np.mean(scores)},
            "individual_scores": {"llm_score": scores}
        }