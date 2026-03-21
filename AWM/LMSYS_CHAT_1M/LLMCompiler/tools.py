"""
工具系统：为ReAct Agent提供各种工具
"""
import json
import re
import random
import time
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from langchain_google_community import GoogleSearchAPIWrapper
from openai import OpenAI, RateLimitError, APIError, BadRequestError
from config import (
    GOOGLE_CSE_API_KEY,
    GOOGLE_CSE_CX,
    SERPER_API_KEY,
    CODE_GENERATOR_API_KEY,
    CODE_GENERATOR_API_BASE,
    CODE_GENERATOR_MODEL,
)
import requests


class Tool:
    """基础工具类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具，返回结果"""
        raise NotImplementedError


# class WebSearchTool(Tool):
#     """网络搜索工具"""
#     
#     def __init__(self):
#         super().__init__(
#             name="web_search",
#             description=(
#                 "Use Google search to look up the information you need. "
#                 "Input: query (str, required) - the text you want to search for; "
#                 "language (str, optional, default='en') - language preference for interpreting results; "
#                 "Example: query='latest AI news', language='en'."
#             )
#         )
#         self._search_client: Optional[GoogleSearchAPIWrapper] = None
#         if GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX:
#             self._search_client = GoogleSearchAPIWrapper(
#                 google_api_key=GOOGLE_CSE_API_KEY,
#                 google_cse_id=GOOGLE_CSE_CX,
#                 k=5,
#             )
#     
#     def execute(
#         self,
#         query: str,
#         language: str = "en",
#         max_results: int = 5,
#         retry_times: int = 9
#     ) -> Dict[str, Any]:
#         """执行Google搜索（基于GoogleSearchAPIWrapper.run）"""
#         if not query or not isinstance(query, str) or not query.strip():
#             return {
#                 "status": "error",
#                 "error": "查询字符串不能为空",
#                 "query": query
#             }
#         if not self._search_client:
#             return {
#                 "status": "error",
#                 "error": "未配置Google搜索API密钥",
#                 "query": query
#             }
#         
#         query = query.strip()
#         try:
#             parsed_max = int(max_results)
#             max_results = max(1, parsed_max)
#         except (TypeError, ValueError):
#             max_results = 5
#         
#         base_delay = 15
#         for attempt in range(retry_times):
#             try:
#                 self._search_client.k = max_results
#                 summary = self._search_client.run(query)
#                 structured_results = []
#                 try:
#                     structured_results = self._search_client.results(query, max_results)
#                 except Exception:
#                     # run() 已成功，结构化结果失败时忽略
#                     structured_results = []
#                 
#                 return {
#                     "status": "success",
#                     "query": query,
#                     "language": language,
#                     "summary": summary.strip() if isinstance(summary, str) else "",
#                     "results": structured_results
#                 }
#                 
#             except Exception as e:
#                 error_msg = str(e)
#                 if attempt < retry_times - 1:
#                     wait = base_delay * (2 ** attempt)
#                     print(f"[WebSearchTool] 错误: {error_msg}. 等待 {wait:.1f}s 后重试（第 {attempt + 1}/{retry_times} 次）...")
#                     time.sleep(wait)
#                 else:
#                     return {
#                         "status": "error",
#                         "error": f"多次重试后仍然失败: {error_msg}",
#                         "query": query,
#                         "retry_attempts": retry_times
#                     }
#         
#         return {
#             "status": "error",
#             "error": "未知错误",
#             "query": query
#         }


class WebSearchTool(Tool):
    """新的网络搜索工具 (使用 Serper API)"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description=(
                "Use web search to look up the information you need. "
                "Input: query (str, required) - the text you want to search for; "
                "language (str, optional, default='en') - language preference for interpreting results; "
                "Example: query='latest AI news', language='en'."
            )
        )
        self.api_key = SERPER_API_KEY
        self.url = "https://google.serper.dev/search"
    
    def execute(
        self,
        query: str,
        language: str = "en",
        max_results: int = 5,
        retry_times: int = 3
    ) -> Dict[str, Any]:
        """执行Serper搜索"""
        if not query or not isinstance(query, str) or not query.strip():
            return {
                "status": "error",
                "error": "查询字符串不能为空",
                "query": query
            }
        
        if not self.api_key or "YOUR_SERPER_API_KEY" in self.api_key:
            return {
                "status": "error",
                "error": "未配置SERPER_API_KEY",
                "query": query
            }
            
        query = query.strip()
        try:
            parsed_max = int(max_results)
            max_results = max(1, parsed_max)
        except (TypeError, ValueError):
            max_results = 5
            
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            "q": query,
            "num": max_results,
            "hl": language
        }
        
        base_delay = 5
        for attempt in range(retry_times):
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # 兼容原有接口，构造 summary 和 structured_results
                organic_results = data.get("organic", [])
                structured_results = []
                summary_parts = []
                
                for item in organic_results:
                    res = {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    }
                    structured_results.append(res)
                    summary_parts.append(f"{res['title']}: {res['snippet']}")
                
                summary = "\n".join(summary_parts)
                
                return {
                    "status": "success",
                    "query": query,
                    "language": language,
                    "summary": summary,
                    "results": structured_results
                }
                
            except Exception as e:
                error_msg = str(e)
                if attempt < retry_times - 1:
                    wait = base_delay * (2 ** attempt)
                    print(f"[WebSearchTool][Serper] 错误: {error_msg}. 等待 {wait:.1f}s 后重试（第 {attempt + 1}/{retry_times} 次）...")
                    time.sleep(wait)
                else:
                    return {
                        "status": "error",
                        "error": f"Serper API 多次重试后仍然失败: {error_msg}",
                        "query": query,
                        "retry_attempts": retry_times
                    }
                    
        return {
            "status": "error",
            "error": "Serper API 未知错误",
            "query": query
        }


class CalculatorTool(Tool):
    """计算器工具"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description=(
                "Perform simple mathematical operations, including +, -, *, / and parentheses. "
                "Input: expression (str, required) - a mathematical expression to evaluate, e.g. '3 * (4 + 5) / 2'."
            )
        )
    
    def execute(self, expression: str) -> Dict[str, Any]:
        """执行数学计算"""
        try:
            # 安全地评估数学表达式
            # 只允许数字、运算符和基本函数
            safe_dict = {
                "__builtins__": {},
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow
            }
            result = eval(expression, safe_dict)
            return {
                "status": "success",
                "expression": expression,
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "expression": expression
            }
class GetCurrentDateTool(Tool):
    """获取当前日期工具"""
    
    def __init__(self):
        super().__init__(
            name="get_current_date",
            description=(
                "Get the current local date and time in the format 'YYYY-MM-DD HH:MM:SS'. "
                "This tool requires NO input parameters. If any input is provided (e.g., 'input'), it will be ignored."
            )
        )
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """获取当前日期和时间
        
        Args:
            **kwargs: 忽略所有输入参数（兼容性处理）
        """
        try:
            now = datetime.now()
            datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
            return {
                "status": "success",
                "datetime": datetime_str
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


class CodeGeneratorTool(Tool):
    """代码生成工具"""
    
    def __init__(self):
        super().__init__(
            name="code_generator",
            description=(
                "Generate, modify, or explain PROGRAMMING CODE ONLY using a code generation model. "
                "ONLY use this tool for actual code-related tasks (writing functions, scripts, algorithms, debugging code, etc.). "
                "DO NOT use this tool for: creative writing, story generation, text descriptions, general Q&A, or any non-programming tasks. "
                "Do NOT call code_generator for formatting or rewriting plain text answers. "
                "Example: prompt='Write a Python function to merge two sorted lists.', language='Python'. "
                "Input: prompt (str, required) - description of the CODE task (generation, modification, or explanation); "
                "language (str, optional) - programming language (e.g. 'Python', 'JavaScript'; "
                "if omitted, the model will infer it from the prompt)."
            )
        )
        if CODE_GENERATOR_API_KEY:
            self.client = OpenAI(
                api_key=CODE_GENERATOR_API_KEY,
                base_url=CODE_GENERATOR_API_BASE
            )
        else:
            self.client = None
        self.model = CODE_GENERATOR_MODEL
    
    def execute(self, prompt: str, language: Optional[str] = None) -> Dict[str, Any]:
        """调用代码生成模型生成代码"""
        if not self.client:
            return {
                "status": "error",
                "error": "未配置CODE_GENERATOR_API_KEY，无法调用代码生成API",
                "prompt": prompt
            }
        
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            return {
                "status": "error",
                "error": "代码生成需求描述不能为空",
                "prompt": prompt
            }
        
        # 构建系统提示词
        system_prompt = "You are an expert code generator. Generate clean, efficient, and well-commented code based on user requirements."
        if language:
            system_prompt += f" Focus on {language} programming language."
        system_prompt += "\n\nCRITICAL OUTPUT REQUIREMENT: Only output the MOST ESSENTIAL and CRITICAL parts of the code. Focus on the core functionality that directly addresses the user's request. Avoid unnecessary boilerplate, verbose comments, or extensive examples unless explicitly requested. Keep your output concise and focused on what matters most."
        
        user_prompt = prompt.strip()
        if language:
            user_prompt = f"Programming language: {language}\n\n{user_prompt}"
        user_prompt += "\n\nIMPORTANT: Only generate the most essential and critical code parts. Focus on the core functionality."
        
        max_retries = 9
        base_delay = 15
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=8192,  # 代码生成可能需要更多token
                    temperature=0.2,  # 代码生成使用较低温度以保证准确性
                    timeout=120  # 代码生成可能需要更长时间
                )
                generated_code = response.choices[0].message.content.strip()
                return {
                    "status": "success",
                    "prompt": prompt,
                    "language": language or "unspecified",
                    "generated_code": generated_code
                }
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt) + random.random()
                    print(f"[CodeGeneratorTool][RateLimit] {e}. 等待 {wait:.1f}s 后重试（第 {attempt + 1}/{max_retries} 次）...")
                    time.sleep(wait)
                else:
                    raise
            except APIError as e:
                if attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)
                    print(f"[CodeGeneratorTool][APIError] {e}. 等待 {wait:.1f}s 后重试（第 {attempt + 1}/{max_retries} 次）...")
                    time.sleep(wait)
                else:
                    raise
            except BadRequestError as e:
                print(f"[CodeGeneratorTool][BadRequest] {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "prompt": prompt
                }
            except Exception as e:
                print(f"[CodeGeneratorTool][Unknown] {type(e)}: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "prompt": prompt
                }
        
        return {
            "status": "error",
            "error": "代码生成API多次重试仍失败",
            "prompt": prompt
            }


class LLMInferenceTool(Tool):
    """模型推理工具（仅供 llmcompile 使用，默认不注册给 ReAct）
    
    根据运行模式选择不同的模型：
    - 产生记忆或GT时：使用GPT4_MODEL
    - mode1-4时：使用QWEN_MODEL
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 9,
        base_delay: float = 3.0,
        retry_backoff: float = 2.0,
    ):
        """
        Args:
            model_name: 模型名称，如果为None则使用QWEN_MODEL
            api_key: API密钥，如果为None则使用QWEN_API_KEY
            base_url: API地址，如果为None则使用QWEN_API_BASE
        """
        # 默认使用QWEN_MODEL（用于mode1-4）
        if model_name is None:
            model_name = QWEN_MODEL
        if api_key is None:
            api_key = QWEN_API_KEY
        if base_url is None:
            base_url = QWEN_API_BASE
        
        super().__init__(
            name="llm_inference",
            description=(
                f"Call {model_name} for REASONING and SUMMARIZATION tasks only. "
                "Use this tool for: answering questions, analyzing information, summarizing content, making inferences, reasoning about problems, etc. "
                "DO NOT use this tool for: generating programming code, writing scripts, or any code-related tasks. "
                "CRITICAL CONTEXT RULES: "
                "(1) This tool does NOT share context with you. "
                "(2) If your task has NO dependencies (no preceding tasks), you MUST explicitly provide the context content as a string in the context parameter (e.g., context=\"<actual context text>\"). DO NOT use placeholder references like \"$1\" when there are no dependencies. "
                "(3) If your task HAS dependencies on previous tasks, you can use context=\"$1\" to reference the output of task 1. "
                "Input: prompt (required) - the main reasoning or summarization task; context (optional) - either explicit context string (if no dependencies) or reference to previous task output like \"$1\" (if has dependencies); max_tokens and temperature are optional. "
            ),
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.model = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_backoff = retry_backoff
    
    def execute(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """调用轻量模型进行推理，带重试机制"""
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            return {
                "status": "error",
                "error": "推理提示词不能为空",
                "prompt": prompt,
            }
        if not self.client:
            return {
                "status": "error",
                "error": "未配置QWEN_API_KEY，无法调用推理API",
                "prompt": prompt,
            }
        
        # 如果有context，将其拼接到prompt中
        prompt = prompt.strip()
        if context and context.strip():
            context_str = context.strip()
            # 拼接context到prompt
            full_prompt = f"{prompt}\n\nContext:\n{context_str}"
        else:
            full_prompt = prompt
        
        # 添加强调：只输出最关键的部分
        full_prompt += "\n\nCRITICAL OUTPUT REQUIREMENT: Only provide the MOST ESSENTIAL and CRITICAL information in your response. Focus on directly answering the question or completing the task. Avoid unnecessary explanations, verbose descriptions, or redundant information. Keep your response concise and focused on what matters most."
        delay = self.base_delay
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw_content = resp.choices[0].message.content.strip()
                clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                
                usage_info = {}
                if hasattr(resp, 'usage') and resp.usage:
                    usage_info = {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                        "total_tokens": getattr(resp.usage, "total_tokens", 0)
                    }
                
                return {
                    "status": "success",
                    "prompt": full_prompt,  # 返回完整的prompt（包含context）
                    "response": clean_content,
                    "usage": usage_info,
                }
            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait = delay + random.random()
                    print(f"[LLMInferenceTool][RateLimit] {e}. 等待 {wait:.1f}s 后重试（第 {attempt + 1}/{self.max_retries} 次）...")
                    time.sleep(wait)
                    delay *= self.retry_backoff
                else:
                    return {"status": "error", "error": f"RateLimit: {e}", "prompt": full_prompt}
            except APIError as e:
                if attempt < self.max_retries - 1:
                    wait = delay
                    print(f"[LLMInferenceTool][APIError] {e}. 等待 {wait:.1f}s 后重试（第 {attempt + 1}/{self.max_retries} 次）...")
                    time.sleep(wait)
                    delay *= self.retry_backoff
                else:
                    return {"status": "error", "error": f"APIError: {e}", "prompt": full_prompt}
            except BadRequestError as e:
                print(f"[LLMInferenceTool][BadRequest] {e}")
                return {"status": "error", "error": str(e), "prompt": full_prompt}
            except Exception as e:
                print(f"[LLMInferenceTool][Unknown] {type(e)}: {e}")
                return {"status": "error", "error": str(e), "prompt": full_prompt}
        
        return {
            "status": "error",
            "error": "推理API多次重试仍失败",
            "prompt": prompt,
        }


class ToolRegistry:
    """工具注册表"""
    
    def __init__(
        self,
        memory_dir: Optional[str] = None,
        enable_memory_tools: bool = False,
        enable_external_tools: bool = True,
    ):
        """
        初始化工具注册表
        
        Args:
            memory_dir: 记忆文件目录（用于Web1和code1工具）
            enable_memory_tools: 是否启用记忆工具（Web1和code1），用于mode3和mode4
            enable_external_tools: 是否启用外部API工具（web_search/code_generator等）
        """
        self.tools: Dict[str, Tool] = {}
        self.memory_dir = memory_dir
        self.enable_memory_tools = enable_memory_tools
        self.enable_external_tools = enable_external_tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        if self.enable_external_tools:
            self.register(WebSearchTool())
        self.register(CalculatorTool())
        self.register(GetCurrentDateTool())
        self.register(LLMInferenceTool())
        if self.enable_external_tools:
            self.register(CodeGeneratorTool())
        
    
    def register(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self.tools.get(name)
    
    def list_tools(self) -> list:
        """列出所有工具"""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools.values()
        ]
    
    def get_tools_description(self) -> str:
        """获取所有工具的描述（用于提示词）"""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)

