import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import math

from llm_client import LLMClient
from tools import ToolRegistry
from embedding_utils import QwenEmbedding

logger = logging.getLogger(__name__)

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
{experience_instruction}

Important things you must follow:
- AVOID CONSECUTIVE REPEATED TOOL CALLS: Do not call the same tool twice in a row with similar input parameters. If you have already called a tool with similar parameters, use the previous result instead of calling it again. If you need to call a tool again, make sure the input parameters are significantly different from the previous call.
- When you decide to finish, do NOT emit any Action (especially not "Action: Final Answer"). Go straight to:
{final_block_example}
- When a problem can be solved using tools, always prioritize using those tools, as they are reliable.
- If you don't get the desired result from the tool, please don't repeat similar questions. If a tool returns an empty result or an error about 'repeated call', do NOT try similar parameters. You should explore further based on your existing knowledge or change your search strategy significantly.
- CRITICAL FORMAT RULES:
    1) Each round of reasoning (excluding the final answer) must output a complete set of four lines in order: Thought -> Action -> Action Input -> Observation. Do not omit Action/Action Input/Observation while only showing Thought.
    2) When you have enough information to answer the question, skip any new Action and go directly to:
{final_block_example_indent}
    3) Action field must be exactly one of [{tool_names}] when used; NEVER use "None", "Final Answer", "final_answer", or any non-tool value in Action.
    4) Do NOT emit an extra "Action:" immediately before the final answer. The final block should be only {final_block_inline}

EXAMPLES OF TOOL USAGE:

Example 1:
User: What is the current stock price of Apple?
Thought: I need to find the current stock price of Apple.
Action: web_search
Action Input: {{"query": "current Apple (AAPL) stock price"}}
Observation: Apple Inc. (AAPL) is trading at $185.92 as of the latest market close.
Thought: I now know the final answer
Final Answer: The current stock price of Apple (AAPL) is $185.92.
{experience_instruction_example1}

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
{experience_instruction_example2}

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
{experience_instruction_example3}

Let`s Begin!
</no_think>
"""

class ReactAgent:
    def __init__(self, llm_client: LLMClient, max_iterations: int = 10, is_edge=False, edge_tool_db=None, cloud_tool_db=None, generate_experience=False):
        self.is_edge = is_edge
        self.edge_tool_db = edge_tool_db
        self.cloud_tool_db = cloud_tool_db
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.generate_experience = generate_experience
        self.registry = ToolRegistry(enable_external_tools=True)
        self.tools = self.registry.tools
        
        # Initialize embedder for semantic similarity checks
        self.embedder = QwenEmbedding()
        
        # Regex patterns from Baseline1
        self.action_pattern = r"Action:\s*(\w+)"
        self.action_input_pattern = r"Action Input:\s*(.*?)(?:\n|$)"
        self.final_answer_pattern = r"Final Answer:\s*(.*?)(?=\s*Experience Summary:|$)"
        self.experience_summary_pattern = r"Experience Summary:\s*(.*)"


    def _construct_prompt(self, user_query: str, history: List[Dict[str, str]], experience: str = None, reference_type: str = "experience") -> List[Dict[str, str]]:
        tool_names = list(self.registry.tools.keys())
        tool_strings = "\n".join([f"{name}: {tool.description}" for name, tool in self.registry.tools.items()])
        
        exp_instr = ""
        if self.generate_experience:
            final_block_example = "  Thought: I now know the final answer\n  Final Answer: ...\n  Experience Summary: ..."
            final_block_example_indent = "       Thought: I now know the final answer\n       Final Answer: ...\n       Experience Summary: ..."
            final_block_inline = '"Thought: I now know the final answer" followed by "Final Answer:" and then "Experience Summary:".'
            exp_instr = (
                "Experience Summary: Summarize the key logic and tool usage of your final workflow.\n"
                "CRITICAL FORMAT REQUIREMENT: You MUST STRICTLY follow this exact formula for the experience summary:\n"
                "If the user purpose is [Intent], then the thought/workflow should be [Actionable steps].\n"
                "Example:\n"
                "Experience Summary: If the user purpose is 'calculating mortgage payments', then the thought/workflow should be 'Use the calculator tool with the interest formula and then provide the monthly breakdown.'"
            )
        else:
            final_block_example = "  Thought: I now know the final answer\n  Final Answer: ..."
            final_block_example_indent = "       Thought: I now know the final answer\n       Final Answer: ..."
            final_block_inline = '"Thought: I now know the final answer" followed by "Final Answer:".'

        system_prompt = STUDENT_SYSTEM_PROMPT_TEMPLATE.format(
            tools=tool_strings,
            tool_names=", ".join(tool_names),
            experience_instruction=exp_instr,
            final_block_example=final_block_example,
            final_block_example_indent=final_block_example_indent,
            final_block_inline=final_block_inline,
            experience_instruction_example1="\nExperience Summary: If the user purpose is 'finding current stock price', then the thought/workflow should be 'Use web_search to find the latest stock price'." if self.generate_experience else "",
            experience_instruction_example2="\nExperience Summary: If the user purpose is 'calculating days to a future date', then the thought/workflow should be 'Use get_current_date to find today, then calculate the difference using calculator'." if self.generate_experience else "",
            experience_instruction_example3="\nExperience Summary: If the user purpose is 'writing a sorting algorithm', then the thought/workflow should be 'Use code_generator to write the algorithm and return it'." if self.generate_experience else ""
        )
        
        # Add reference guidance based on similarity level if experience provided
        if experience:
            if reference_type == "trace":
                reference_instruction = (
                    "The system has detected that this user intent has occurred before with VERY HIGH similarity. "
                    "Please refer to the following RECORDED TOOL CALL TRACE as a guide. "
                    "Based on these recorded tool calls, decide whether you need to call additional tools to gather more information "
                    "from other perspectives. If the tools in this trace have already addressed the query comprehensively, "
                    "you should provide the final answer directly based on the logic of the Final Answer in the trace."
                    "CRITICAL: If the entities (people, places, things) in the current query are DIFFERENT from those in the experience, you MUST use tools to gather information for the NEW entities. "
                    "Do NOT directly provide a final answer based on old data if the entities have changed; instead, replicate the workflow for the current parameters."
                )
            else:
                reference_instruction = (
                    "The system has detected that this user intent has occurred before. "
                    "Please refer to the following abstracted experience as a guide "
                    "and use tools under this guidance to address the current query."
                    "CRITICAL: If the entities (people, places, things) in the current query are DIFFERENT from those in the experience, you MUST use tools to gather information for the NEW entities. "
                    "Do NOT directly provide a final answer based on old data if the entities have changed; instead, replicate the workflow for the current parameters."
                )


            system_prompt += (
                f"\n\n[REFERENCE: SIMILAR PAST EXPERIENCE]\n"
                f"{reference_instruction}\n"
                f"- Experience:\n{experience}\n"
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
        """解析模型输出 (From Baseline1)"""
        # 1. 核心过滤逻辑：移除 <think> 标签及其内容
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # 2. 优先尝试解析 Final Answer 和 Experience Summary
        final_match = re.search(self.final_answer_pattern, text, re.DOTALL)
        if final_match:
            answer = final_match.group(1).strip()
            exp_match = re.search(self.experience_summary_pattern, text, re.DOTALL)
            experience = exp_match.group(1).strip() if exp_match else None
            return "final", {"answer": answer, "experience": experience}

        # 3. 尝试解析 Action 和 Action Input
        action_match = re.search(self.action_pattern, text)
        action_input_match = re.search(self.action_input_pattern, text)

        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            return "action", (action, action_input)

        return "unknown", text

    def _execute_tool(self, action: str, action_input_str: str, stats: Dict[str, Any], transfer_stats: Optional[Dict[str, int]] = None) -> str:
        from config import TOOL_SIMILARITY_THRESHOLD, TOOL_MEMORY_LIBRARY
        
        # 1. 创新点2 - 工具记忆库 (Tool Memory Library)
        if TOOL_MEMORY_LIBRARY and action in ["web_search", "code_generator"]:
            if getattr(self, "is_edge", False):
                if getattr(self, "edge_tool_db", None):
                    found, output = self.edge_tool_db.search(action, action_input_str, TOOL_SIMILARITY_THRESHOLD)
                    if found: 
                        print(f"--- 工具调用被端侧工具记忆库拦截: {action} ---")
                        return output
                if getattr(self, "cloud_tool_db", None):
                    # 记录端云通信传输量: 使用原始字符串 (action + input)
                    if transfer_stats is not None:
                        transfer_stats["s2l"] += len(action.encode('utf-8')) + len(action_input_str.encode('utf-8'))
                    
                    if getattr(self, "edge_tool_db", None):
                        query_emb = self.edge_tool_db.encoder.encode_single(action_input_str)
                    else:
                        query_emb = self.cloud_tool_db.encoder.encode_single(action_input_str)
                    found, output = self.cloud_tool_db.search_by_emb(action, query_emb, TOOL_SIMILARITY_THRESHOLD)
                    if found:
                        print(f"--- 工具调用被云侧工具记忆库拦截 (端侧模式): {action} ---")
                        if transfer_stats is not None:
                            transfer_stats["l2s"] += len(output.encode('utf-8'))
                        return output
            else:
                if getattr(self, "cloud_tool_db", None):
                    found, output = self.cloud_tool_db.search(action, action_input_str, TOOL_SIMILARITY_THRESHOLD)
                    if found: 
                        print(f"--- 工具调用被云侧工具记忆库拦截: {action} ---")
                        return output

        # 2. 标准工具执行
        if action not in self.tools:
            return f"Error: Tool {action} not found."
            
        if action == "web_search":
            stats["web_search_calls"] += 1
        
        try:
            # 参数解析 (Aligned with Baseline1 robust parsing)
            cleaned_input = action_input_str.strip().strip("'").strip('"')
            try:
                params = json.loads(cleaned_input)
            except Exception:
                if action == "calculator":
                    params = {"expression": action_input_str}
                elif action == "web_search":
                    params = {"query": action_input_str}
                elif action == "code_generator":
                    params = {"prompt": action_input_str}
                else:
                    params = {"input": action_input_str}

            # 执行工具
            observation_dict = {}
            if action == "web_search":
                observation_dict = self.tools["web_search"].execute(query=params.get("query", action_input_str))
            elif action == "calculator":
                observation_dict = self.tools["calculator"].execute(expression=params.get("expression", action_input_str))
            elif action == "get_current_date":
                observation_dict = self.tools["get_current_date"].execute()
            elif action == "code_generator":
                # Token estimation fallback (Baseline1 logic)
                prompt_text = params.get("prompt", action_input_str)
                input_tokens_est = int(len(str(prompt_text).split()) * 1.3)
                
                observation_dict = self.tools["code_generator"].execute(
                    prompt=prompt_text,
                    language=params.get("language", None)
                )
                
                # Update tokens: Prioritize real usage from tool result if available
                p_tokens = 0
                c_tokens = 0
                if observation_dict.get("status") == "success":
                    usage = observation_dict.get("usage") or observation_dict.get("token_usage")
                    if isinstance(usage, dict):
                        p_tokens = usage.get("prompt_tokens", 0)
                        c_tokens = usage.get("completion_tokens", 0)
                
                if p_tokens > 0 or c_tokens > 0:
                    stats["code_input_tokens"] += p_tokens
                    stats["code_output_tokens"] += c_tokens
                else:
                    # Fallback to estimate
                    output_tokens_est = int(len(str(observation_dict.get("result", "")).split()) * 1.3)
                    stats["code_input_tokens"] += input_tokens_est
                    stats["code_output_tokens"] += output_tokens_est
            
            output_str = json.dumps(observation_dict, ensure_ascii=False)
            
            # 3. 创新点2 - 工具记忆库库更新: 仅在结果非空且执行成功时存储
            is_valid_output = output_str.strip() != "{}" and output_str.strip() != ""
            
            # 显式根据工具状态判断：如果标记为 success 且如果有结果列表不能全为空
            status = observation_dict.get('status')
            if status == 'success':
                if 'result' in observation_dict and not observation_dict.get('result'): 
                    is_valid_output = False
            elif status in ('failed', 'error'):
                is_valid_output = False

            if TOOL_MEMORY_LIBRARY and action in ["web_search", "code_generator"] and is_valid_output:
                if getattr(self, "is_edge", False) and getattr(self, "edge_tool_db", None):
                    self.edge_tool_db.add_record(action, action_input_str, output_str)
                elif not getattr(self, "is_edge", False) and getattr(self, "cloud_tool_db", None):
                    self.cloud_tool_db.add_record(action, action_input_str, output_str)
                
            return output_str

        except Exception as e:
            err_dict = {"status": "error", "message": f"Error executing tool {action}: {str(e)}"}
            return json.dumps(err_dict, ensure_ascii=False)

    def run(self, user_query: str, history: List[Dict[str, str]] = [], experience: str = None, reference_type: str = "experience") -> Dict[str, Any]:
        """执行 ReAct 循环 (Aligned with Baseline1 structure)"""
        # 端↔云通信统计（以UTF-8字节数精确记录）
        transfer_stats = {"s2l": 0, "l2s": 0}
        
        # Initial transfer estimate: only count if sending from edge to cloud
        if not self.is_edge:
            try:
                # We assume the system prompt and experience knowledge are handled/pre-positioned 
                # and only the core interactive context is "transferred"
                payload = {
                    "user_query": user_query,
                    "history": history
                }
                serialized = json.dumps(payload, ensure_ascii=False)
                transfer_stats["s2l"] += len(serialized.encode('utf-8'))
            except Exception:
                transfer_stats["s2l"] += len(str(user_query).encode('utf-8')) + len(str(history).encode('utf-8'))

        messages = self._construct_prompt(user_query, history, experience, reference_type)
        trace = []
        current_step = 0
        final_answer = ""

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_stats = {"web_search_calls": 0, "code_input_tokens": 0, "code_output_tokens": 0}

        # Track tool input embeddings for semantic deduplication
        tool_input_embeddings: Dict[str, list] = {}

        while current_step < self.max_iterations:
            response = self.llm.generate(messages, stop=["Observation:"])
            
            usage = response.get("usage", {})
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)

            content = response.get("content", "")
            if "<think>" in content:
                content = re.sub(r'<think>.*?(?:</think>|$)', '', content, flags=re.DOTALL).strip()
            print('模型的输出',content)
            msg_type, data = self.parse_output(content)
            
            if msg_type == "final":
                final_answer = data["answer"]
                experience_summary = data.get("experience")
                # 统计 trace：记录 Thought(在content中) 和 Final Answer
                trace.append({
                    "step": current_step, 
                    "type": "final",
                    "thought": content, 
                    "final_answer": final_answer,
                    "experience_summary": experience_summary
                })
                break

            if msg_type == "action":
                action, action_input = data
                # 统计 trace：记录 Thought(在content中), Action 以及 Action Input
                trace.append({
                    "step": current_step, 
                    "type": "action",
                    "action": action,
                    "action_input": action_input,
                    "thought": content
                })

                # ===== 工具调用去重逻辑 (From Baseline1) =====
                skip_tool_call = False
                similar_threshold = 1.0
                try:
                    cur_emb = self.embedder.encode_single(str(action_input))
                    if cur_emb.size > 0:
                        prev_emb_list = tool_input_embeddings.get(action, [])
                        for prev_emb in prev_emb_list:
                            denom = (np.linalg.norm(cur_emb) * np.linalg.norm(prev_emb) + 1e-8)
                            sim = float(np.dot(cur_emb, prev_emb) / denom)
                            if sim > similar_threshold:
                                skip_tool_call = True
                                break
                        if not skip_tool_call:
                            tool_input_embeddings.setdefault(action, []).append(cur_emb)
                except Exception:
                    skip_tool_call = False

                if skip_tool_call:
                    observation_val = {
                        "status": "error",
                        "message": (
                            f"Note: A similar call to '{action}' was already performed with nearly identical parameters. "
                            "To prevent redundant operations and potential infinite loops, the system has blocked this call. "
                            "Please either significantly change your search parameters or use your current information to derive a Final Answer."
                        )
                    }
                    observation = json.dumps(observation_val, ensure_ascii=False)
                else:
                    observation = self._execute_tool(action, action_input, stats=tool_stats, transfer_stats=transfer_stats)

                obs_text = f"Observation: {observation}"
                print('工具的返回结果', obs_text[:200] + '...' if len(obs_text) > 200 else obs_text)
                trace.append({"step": current_step, "observation": observation, "type": "observation"})
                
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": obs_text})
            else:
                final_answer = content
                break
                
            current_step += 1
            
        if not self.is_edge:
            transfer_stats["l2s"] += len(str(final_answer).encode('utf-8')) if final_answer else 0
        return {
            "answer": final_answer,
            "experience_summary": experience_summary if 'experience_summary' in locals() else None,
            "trace": trace,
            "usage": total_usage,
            "total_calls": current_step + 1, 
            "total_steps": current_step + 1,
            "search_calls": tool_stats.get("web_search_calls", 0),
            "code_input_tokens": tool_stats.get("code_input_tokens", 0),
            "code_output_tokens": tool_stats.get("code_output_tokens", 0),
            "transfer_stats": transfer_stats
        }

