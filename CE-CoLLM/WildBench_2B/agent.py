import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import math

# 假设你的 tools.py 在同级目录
from tools import ToolRegistry
from embedding_utils import QwenEmbedding

logger = logging.getLogger(__name__)

def total_logprob(logprob_list):
    return sum(logprob_list)

# 定义更严谨的系统提示词模板
STUDENT_SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant skilled at using tools. 

CRITICAL: TOOLS ARE ESSENTIAL FOR HIGH-QUALITY ANSWERS. You MUST actively use tools to gather accurate, up-to-date information and provide comprehensive answers. Using tools significantly improves answer quality, accuracy, and completeness.

Available tools:
{tools}

TOOL USAGE GUIDELINES:
- ACTIVELY USE TOOLS: Whenever you need information, verification, calculations, or any data that tools can provide, USE THE TOOLS. Do not rely solely on your internal knowledge when tools are available.
- USE MULTIPLE TOOLS: For complex questions, use multiple tools in sequence to gather comprehensive information. Each tool can provide different perspectives and data.
- VERIFY WITH TOOLS: When uncertain about facts, numbers, dates, or current information, use tools to verify and get accurate data.
- TOOL-FIRST APPROACH: Prioritize using tools over guessing. If a tool can help answer the question or provide supporting information, use it.

Strictly use the following format to think about how to respond to the conversation:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this can repeat N times, N>=0. Use tools multiple times as needed to gather comprehensive information.)

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important things you must follow:
- AVOID CONSECUTIVE REPEATED TOOL CALLS: Do not call the same tool twice in a row with similar input parameters. If you have already called a tool with similar parameters, use the previous result instead of calling it again. If you need to call a tool again, make sure the input parameters are significantly different from the previous call.
- When you decide to finish, do NOT emit any Action (especially not "Action: Final Answer"). Go straight to:
  Thought: I now know the final answer
  Final Answer: ...
- When a problem can be solved using tools, always prioritize using those tools, as they are reliable.
- If you don't get the desired result from the tool, please don't repeat similar questions. You should explore further based on your existing knowledge. Do not repeatedly call tools with the same name and similar input parameters in the same conversation.
- CRITICAL FORMAT RULES:
    1) Each round of reasoning (excluding the final answer) must output a complete set of four lines in order: Thought -> Action -> Action Input -> Observation. Do not omit Action/Action Input/Observation while only showing Thought.
    2) When you have enough information to answer the question, skip any new Action and go directly to:
       Thought: I now know the final answer
       Final Answer: ...
    3) Action field must be exactly one of [{tool_names}] when used; NEVER use "None", "Final Answer", "final_answer", or any non-tool value in Action.
    4) Do NOT emit an extra "Action:" immediately before the final answer. The final block should be only "Thought: I now know the final answer" followed by "Final Answer:".

EXAMPLES OF TOOL USAGE:

Example 1:
User: What is the current stock price of Apple?
Thought: I need to find the current stock price of Apple.
Action: web_search
Action Input: {{"query": "current Apple (AAPL) stock price"}}
Observation: Apple Inc. (AAPL) is trading at $185.92 as of the latest market close.
Thought: I now know the final answer
Final Answer: The current stock price of Apple (AAPL) is $185.92.

Example 2:
User: How many days are there until Christmas 2026?
Thought: I need to know today's date and the date of Christmas 2026 to calculate the difference.
Action: get_current_date
Action Input: {{}}
Observation: 2026-04-09 19:31:19
Thought: Today is April 9, 2026. Christmas 2026 is December 25, 2026. I should calculate the number of days between these dates.
Action: calculator
Action Input: {{"expression": "21 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 25"}}
Observation: 260
Thought: I now know the final answer
Final Answer: There are approximately 260 days until Christmas 2026 from today (April 9, 2026).

Example 3:
User: Write a bubble sort algorithm in Python.
Thought: I need to generate a Python implementation of the bubble sort algorithm.
Action: code_generator
Action Input: {{"prompt": "write a bubble sort algorithm in python", "language": "Python"}}
Observation: ```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```
Thought: I now know the final answer
Final Answer: Here is a standard implementation of the Bubble Sort algorithm in Python:

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```

Let`s Begin!
</no_think>
"""

class ReactAgent:
    def __init__(self, llm_client: Any, max_iterations: int = 10):
        """
        初始化 ReAct Agent
        
        Args:
            llm_client: 必须包含 .generate(messages, stop, logprobs) 方法的客户端
            max_iterations: 最大迭代次数
        """
        self.llm = llm_client
        self.max_iterations = max_iterations
        
        # 初始化你提供的工具注册表
        self.registry = ToolRegistry(enable_external_tools=True)

        # 初始化嵌入模型，用于判断工具调用输入的语义相似度
        self.embedder = QwenEmbedding()
        
        # 预编译更稳健的正则表达式
        self.action_pattern = r"Action:\s*(\w+)"
        # 修改后：允许 Action Input 为空
        self.action_input_pattern = r"Action Input:\s*(.*?)(?:\n|$)"
        self.final_answer_pattern = r"Final Answer:\s*(.*)"


    def _construct_prompt(self, user_query: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        tool_names = list(self.registry.tools.keys())
        tool_strings = "\n".join([f"{name}: {tool.description}" for name, tool in self.registry.tools.items()])
        system_prompt = STUDENT_SYSTEM_PROMPT_TEMPLATE.format(
            tools=tool_strings,
            tool_names=", ".join(tool_names)
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            # 添加历史对话记录的提示信息，帮助模型区分背景和当前问题
            messages.append({
                "role": "user", 
                "content": "The following are historical dialogue records provided to help you understand the context. Please focus on answering the current question presented at the end."
            })
            for msg in history:
                messages.append(msg)
            
        messages.append({"role": "user", "content": f"Current Question to Answer: {user_query}"})
        return messages

    def parse_output(self, text: str) -> Tuple[str, Any]:
        """
        解析模型输出。
        返回: (type, data) 
        type: 'final', 'action', 或 'unknown'
        """
        # 1. 核心过滤逻辑：移除 <think> 标签及其内容
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # 2. 优先尝试解析 Final Answer (支持多行)
        final_match = re.search(self.final_answer_pattern, text, re.DOTALL)
        if final_match:
            return "final", final_match.group(1).strip()

        # 3. 尝试解析 Action 和 Action Input
        action_match = re.search(self.action_pattern, text)
        action_input_match = re.search(self.action_input_pattern, text)

        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            return "action", (action, action_input)

        return "unknown", text

    def _execute_tool(self, action: str, action_input_str: str, stats: Dict[str, Any]) -> str:
        tool = self.registry.get_tool(action)
        if not tool:
            return f"Error: Tool '{action}' not found in registry."

        # 1. 统计搜索调用次数
        if action == "web_search":
            stats["web_search_calls"] += 1
        
        # 2. 统计时间工具调用 (如果需要)
        if action == "get_current_date":
            stats["get_current_date_calls"] += 1

        try:
            # 参数解析逻辑保持不变，同时为代码生成工具统计输入 Token
            cleaned_input = action_input_str.strip().strip("'").strip('"')
            try:
                params = json.loads(cleaned_input)
            except Exception:
                if action == "calculator":
                    params = {"expression": action_input_str}
                elif action in ["web_search"]:
                    params = {"query": action_input_str}
                elif action in ["code_generator"]:
                    params = {"prompt": action_input_str}
                else:
                    params = {"input": action_input_str}

            # 针对代码生成工具，先根据 prompt 估算输入 Token（作为兜底估计）
            if action == "code_generator":
                prompt_text = params.get("prompt", action_input_str)
                input_tokens_est = len(str(prompt_text).split()) * 1.3
            else:
                input_tokens_est = 0

            # 执行工具
            result = tool.execute(**params)
            
            # 3. 统计代码生成工具的输入 + 输出 Token 数：
            #    优先使用工具返回的准确 usage；如果没有，则回退到基于文本长度的估算。
            if action == "code_generator" and isinstance(result, dict):
                tokens_from_api = 0
                # 优先从 result 中读取 usage 或 token_usage
                usage = result.get("usage") or result.get("token_usage")
                if isinstance(usage, dict):
                    if "total_tokens" in usage:
                        tokens_from_api = usage.get("total_tokens", 0)
                    else:
                        tokens_from_api = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

                if tokens_from_api > 0:
                    # 有准确的 Token 用量，直接使用
                    stats["code_tokens"] += int(tokens_from_api)
                else:
                    # 否则退回到基于文本长度的估算
                    # 假设你的 tools.py 的 CodeGeneratorTool 结果里包含 generated_code
                    code_text = result.get("generated_code", "")
                    # 粗略估计输出 Token（单词数 * 1.3）
                    output_tokens_est = len(code_text.split()) * 1.3
                    stats["code_tokens"] += int(input_tokens_est + output_tokens_est)

            # 结果返回逻辑保持不变...
            if isinstance(result, dict):
                if result.get("status") == "success":
                    return str(result.get("result") or result.get("summary") or result.get("generated_code") or result.get("datetime") or result)
                return json.dumps(result, ensure_ascii=False)
            return str(result)

        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def run(self, user_query: str, history: List[Dict[str, str]] = []) -> Dict[str, Any]:
        """执行 ReAct 循环，并统计端云通信量（字符数近似）。"""
        # 端↔云通信统计（以UTF-8字节数精确记录）
        transfer_stats = {
            "s2l": 0,
            "l2s": 0,
        }
        # 如果端云有通信，则只会在初始阶段以及返回结果阶段发生。
        # 这里仅统计用户发送的 query 和 history，系统提示词通常预置在云端，不计入传输量。
        try:
            payload = {"query": user_query, "history": history}
            serialized = json.dumps(payload, ensure_ascii=False)
            transfer_stats["s2l"] += len(serialized.encode('utf-8'))
        except Exception:
            transfer_stats["s2l"] += len(str(user_query).encode('utf-8')) + len(str(history).encode('utf-8'))

        messages = self._construct_prompt(user_query, history)
        trace = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        current_step = 0
        final_answer = ""
        confidence = 0.0

        # 工具使用统计：搜索调用次数 + 代码生成输入/输出 Token 估计
        tool_stats: Dict[str, Any] = {
            "web_search_calls": 0,
            "get_current_date_calls": 0,
            "code_tokens": 0,  # 代码生成工具的输入 + 输出 Token 近似和
        }

        # 记录每种工具在当前 Query 下已经使用过的输入嵌入，用于避免“语义上高度重复”的调用
        # 结构：{tool_name: [embedding_vector, ...]}
        tool_input_embeddings: Dict[str, list] = {}

        while current_step < self.max_iterations:
            # 调用 LLM
            response = self.llm.generate(messages, stop=["Observation:"], logprobs=True)
            
            # 累计 Usage
            usage = response.get("usage", {})
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)

            content = response.get("content", "")
            if "<think>" in content:
                content = re.sub(r'<think>.*?(?:</think>|$)', '', content, flags=re.DOTALL).strip()

            # 从 llm_client 客户端直接获取已经计算完毕的置信度
            step_confidence = response.get("confidence", 0.0)

            print(f'模型的输出：{content}')
            # 解析输出
            msg_type, data = self.parse_output(content)
            
            if msg_type == "final":
                final_answer = data
                confidence = step_confidence
                trace.append({"step": current_step, "thought": content, "type": "final"})
                break

            if msg_type == "action":
                action, action_input = data
                trace.append({"step": current_step, "thought": content, "type": "action", "action": action})

                # ===== 工具调用去重逻辑（基于语义相似度）=====
                # 对同一工具名下的多次调用，如果输入语义相似度 > 0.8，则不再真正调用工具，
                # 而是提示模型根据现有知识直接回答。
                skip_tool_call = False
                similar_threshold = 0.8

                try:
                    # 计算当前 action_input 的嵌入
                    cur_emb = self.embedder.encode_single(str(action_input))
                    if cur_emb.size > 0:
                        prev_emb_list = tool_input_embeddings.get(action, [])
                        for prev_emb in prev_emb_list:
                            # 余弦相似度
                            denom = (np.linalg.norm(cur_emb) * np.linalg.norm(prev_emb) + 1e-8)
                            sim = float(np.dot(cur_emb, prev_emb) / denom)
                            if sim > similar_threshold:
                                skip_tool_call = True
                                break

                        # 如果这次调用不会被跳过，则把嵌入记录下来，供后续比较使用
                        if not skip_tool_call:
                            tool_input_embeddings.setdefault(action, []).append(cur_emb)
                except Exception as _e:
                    # 嵌入或相似度计算失败时，不影响正常工具调用流程
                    skip_tool_call = False

                if skip_tool_call:
                    # 按照 LMSYS ReAct 实现中的格式返回结构化错误信息（再由外层作为 Observation 文本）
                    error_message = {
                        "status": "error",
                        "error": "Please do not repeatedly call similar tools; answer this conversation based on existing knowledge.",
                        "tool_name": action,
                        # 这里无法拿到真实 kwargs，就把原始 Action Input 包一层，便于后续调试或分析
                        "tool_input": {"raw_input": action_input},
                        "message": (
                            "Detected repeated calls to the same tool within one query, "
                            "with input parameters having a similarity greater than 0.8. "
                            "Please use the result from the previous tool call, or ensure "
                            "that the new call parameters are significantly different."
                        ),
                    }
                    observation = json.dumps(error_message, ensure_ascii=False)
                else:
                    # 执行工具
                    observation = self._execute_tool(action, action_input, stats=tool_stats)

                obs_text = f"Observation: {observation}"
                print(obs_text[:100] + "...")
                # 更新对话历史
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": obs_text})
                
                trace.append({"step": current_step, "observation": observation, "type": "observation"})
            else:
                # 解析失败，视为最终答案或报错
                final_answer = content
                break

            current_step += 1
        # 统计云端返回给端侧的内容大小（以UTF-8字节数精确记录网络下行量）
        transfer_stats["l2s"] += len(str(final_answer).encode('utf-8')) if final_answer else 0
        return {
            "answer": final_answer,
            "trace": trace,
            "confidence": confidence,
            "usage": total_usage,
            "total_steps": current_step + 1,
            "transfer_stats": transfer_stats,
            # 供 main.py 汇总使用的统计字段
            "search_calls": tool_stats.get("web_search_calls", 0),
            "code_tokens": tool_stats.get("code_tokens", 0),
        }
