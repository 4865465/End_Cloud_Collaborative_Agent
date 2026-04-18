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
import requests
from langchain_community.utilities import GoogleSerperAPIWrapper
from openai import OpenAI, RateLimitError, APIError, BadRequestError
from config import (
    GOOGLE_CSE_API_KEY,
    GOOGLE_CSE_CX,
    SERPER_API_KEY,
    CODE_GENERATOR_API_KEY,
    CODE_GENERATOR_API_BASE,
    CODE_GENERATOR_MODEL,
)


class Tool:
    """基础工具类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具，返回结果"""
        raise NotImplementedError

class WebSearchTool(Tool):
    """网络搜索工具"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description=(
                "Use Serper (Google search API) to look up the information you need. "
                "Input: query (str, required) - the text you want to search for; "
                "language (str, optional, default='en') - language preference for interpreting results; "
                "Example: query='latest AI news', language='en'."
            )
        )
        self._search_client: Optional[GoogleSerperAPIWrapper] = None
        if SERPER_API_KEY:
            self._search_client = GoogleSerperAPIWrapper(
                serper_api_key=SERPER_API_KEY,
                k=5,
            )
    
    def execute(
        self,
        query: str,
        language: str = "en",
        max_results: int = 5,
        retry_times: int = 9
    ) -> Dict[str, Any]:
        """执行Serper搜索"""
        if not query or not isinstance(query, str) or not query.strip():
            return {
                "status": "error",
                "error": "查询字符串不能为空",
                "query": query
            }
        if not self._search_client:
            return {
                "status": "error",
                "error": "未配置Serper API密钥 (SERPER_API_KEY)",
                "query": query
            }
        
        query = query.strip()
        try:
            parsed_max = int(max_results)
            max_results = max(1, parsed_max)
        except (TypeError, ValueError):
            max_results = 5
        
        base_delay = 15
        for attempt in range(retry_times):
            try:
                self._search_client.k = max_results
                summary = self._search_client.run(query)
                structured_results = []
                try:
                    structured_results = self._search_client.results(query, max_results)
                except Exception:
                    # run() 已成功，结构化结果失败时忽略
                    structured_results = []
                
                return {
                    "status": "success",
                    "query": query,
                    "language": language,
                    "summary": summary.strip() if isinstance(summary, str) else "",
                    "results": structured_results
                }
                
            except Exception as e:
                error_msg = str(e)
                if attempt < retry_times - 1:
                    wait = base_delay * (2 ** attempt)
                    print(f"[WebSearchTool] 错误: {error_msg}. 等待 {wait:.1f}s 后重试（第 {attempt + 1}/{retry_times} 次）...")
                    time.sleep(wait)
                else:
                    return {
                        "status": "error",
                        "error": f"多次重试后仍然失败: {error_msg}",
                        "query": query,
                        "retry_attempts": retry_times
                    }
        
        return {
            "status": "error",
            "error": "未知错误",
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
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
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
                "result": datetime_str
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
                "error": "代码生成需求描述不能为空"
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
                    "result": generated_code
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
                    "error": str(e)
                }
            except Exception as e:
                print(f"[CodeGeneratorTool][Unknown] {type(e)}: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        return {
            "status": "error",
            "error": "代码生成API多次重试仍失败"
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

