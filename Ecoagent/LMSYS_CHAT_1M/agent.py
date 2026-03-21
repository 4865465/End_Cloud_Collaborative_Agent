import json
import re
import time
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, wait
import traceback

from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

from output_parser import LLMCompilerPlanParser, Task
from tools import WebSearchTool, CalculatorTool, GetCurrentDateTool, CodeGeneratorTool, LLMInferenceTool

class State(TypedDict):
    messages: Annotated[list, add_messages]
    planned_tasks: list
    replan_count: int

def clean_output(content: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return cleaned

class LLMCompileAgent:
    PROMPT_TEMPLATE = """
================================ System Message ================================

Given a user query, create a plan to solve it with the utmost parallelizability. Each plan should comprise an action from the following {num_tools} types:
{tool_descriptions}
{num_tools}. join(): Collects and combines results from prior actions.

CRITICAL PLAN LIMITATION - READ THIS CAREFULLY:
 - You MUST ensure that each Plan contains NO MORE THAN 3 tasks (excluding the final join() call)!!!
 - If you think you need more than 3 tasks, you MUST break the problem into smaller parts and use replanning.
 - This limitation helps you focus on the most essential actions and provide concise, effective solutions.

 - An LLM agent is called upon invoking join() to either finalize the user query or wait until the plans are executed.
 - join should always be the last action in the plan, and will be called in two scenarios:
   (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.
   (b) if the answer cannot be determined in the planning phase before you execute the plans. 
Guidelines:
  - Each action MUST have a unique ID, starting from 1 and strictly increasing (1, 2, 3... no skipping).
  - REMINDER: Maximum 3 tasks per plan (excluding join()). Count carefully: 1, 2, 3, then join().
  - Inputs for actions can either be constants or outputs from preceding actions (e.g., $1).
  - CRITICAL: A task CANNOT depend on itself. You cannot use $id to reference the same task.
  - CRITICAL: Every plan MUST end with join()<END_OF_PLAN>
  - (Optional) You can add a 'Thought:' line before a tool call to explain the strategy.

CRITICAL TOOL USAGE:
  - web_search(query="query"): For uncertain questions or latest information.
  - code_generator(prompt="...", language="python"): ONLY for programming/algorithms.
  - llm_inference(prompt="...", context="$1"): ONLY for reasoning and summarization.
  - calculator(expression="..."): For mathematical calculations.
  - join(): Always the last action to finalize the answer.

CRITICAL FORMATTING:
  - DO NOT use wrappers like "tool_input=" or "args=". Use direct parameters: tool_name(param="value").
  - All string parameters MUST be in quotes. Task format: "ID. tool_name(arguments)"
  - CRITICAL: Maximum 3 tasks per plan (plus join()).
  - DO NOT skip task IDs or use non-integer IDs. IDs must be consecutive: 1, 2, 3...
  - CRITICAL: Every plan MUST end with join()<END_OF_PLAN>.
  - Only use the provided tool names exactly as listed above (case-sensitive).

============================= Messages Placeholder =============================

{replan}

{messages}

================================ System Message ================================

Remember, ONLY respond with the task list in the correct format! 

Example format (CORRECT):
1. web_search(query="latest AI developments")
Thought: I need to summarize the search results
2. llm_inference(prompt="Summarize the key points from the search", context="$1")
3. join()<END_OF_PLAN>

WRONG examples (DO NOT use these formats):
- llm_inference(tool_input="prompt=\\"...\\"")  ❌ WRONG: Don't use tool_input wrapper
- web_search(input="query=\\"...\\"")  ❌ WRONG: Don't use input wrapper
- code_generator(args="prompt=\\"...\\"")  ❌ WRONG: Don't use args wrapper

IMPORTANT: 
- Ensure every task follows the exact format above.
- CRITICAL: The plan MUST end with join()<END_OF_PLAN>.
- Invalid or malformed tasks will cause errors.
</no_think>
""".strip()

    def __init__(self, plan_llm, exec_llm, max_iterations: int = 3):
        self.plan_llm = plan_llm
        self.exec_llm = exec_llm
        self.max_iterations = max_iterations
        self._replan_count = 0
        self._plan_round_counter = 0

        self.search_tool = WebSearchTool()
        self.calculator_tool = CalculatorTool()
        self.date_tool = GetCurrentDateTool()
        self.code_tool = CodeGeneratorTool()
        self.llm_infer_tool = LLMInferenceTool(
            model_name=exec_llm.model_name,
            api_key=exec_llm.api_key,
            base_url=exec_llm.base_url
        )

        class _WebSearchInput(BaseModel):
            query: str = Field(description="Search keywords")

        class _CodeGenInput(BaseModel):
            prompt: str = Field(description="Code generation request")
            language: Optional[str] = Field(default=None, description="Language (optional)")

        class _CalculatorInput(BaseModel):
            expression: str = Field(description="Mathematical expression, e.g. 3*(4+5)/2")

        class _EmptyInput(BaseModel):
            pass

        class _LLMInferenceInput(BaseModel):
            prompt: str = Field(description="Prompt for inference")
            context: Optional[str] = Field(default=None, description="Context information, usually output from preceding tasks")

        self.tools = [
            StructuredTool.from_function(func=self.search_tool.execute, name=self.search_tool.name, description=self.search_tool.description, args_schema=_WebSearchInput),
            StructuredTool.from_function(func=self.calculator_tool.execute, name=self.calculator_tool.name, description=self.calculator_tool.description, args_schema=_CalculatorInput),
            StructuredTool.from_function(func=self.date_tool.execute, name=self.date_tool.name, description=self.date_tool.description, args_schema=_EmptyInput),
            StructuredTool.from_function(func=self.code_tool.execute, name=self.code_tool.name, description=self.code_tool.description, args_schema=_CodeGenInput),
            StructuredTool.from_function(func=self.llm_infer_tool.execute, name=self.llm_infer_tool.name, description=self.llm_infer_tool.description, args_schema=_LLMInferenceInput)
        ]

        self.parser = LLMCompilerPlanParser(tools=self.tools)
        
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.large_calls = 0
        self.small_calls = 0

        
        self.chain = self._create_graph()

    def _compress_observation(self, observation: str) -> str:
        # If observation is short, don't waste tokens/time compressing it
        if not observation or len(observation.strip()) <= 150:
            return observation.strip()

        prompt = (
            "You are an expert at information distillation. You are provided with execution results from one or more tasks. "
            "Your goal is to compress these results into a concise summary for high-level reasoning, significantly reducing token usage while preserving all vital information.\n\n"
            "CRITICAL GUIDELINES:\n"
            "1. PRESERVE all key data: specific numbers, dates, names, facts, and essential URLs.\n"
            "2. DATA STRUCTURES: If the output is a list or table, summarize the key entries rather than listing everything.\n"
            "3. CODE/LOGIC: If code or algorithms were produced, keep the core logic/result but remove boilerplate.\n"
            "4. WEB SEARCH: Extract only the most relevant snippets that help answer a query.\n"
            "5. NO FILLER: Remove polite phrases, redundant explanations, or meta-commentary from the tool.\n"
            "6. STRUCTURE: Maintain the 'Task ID [Tool] (Args)' header format for each task, summarizing the results under each respective header.\n\n"
            f"Observation Content to Compress:\n{observation}"
        )
        res = self.exec_llm.generate([{"role": "user", "content": prompt}], temperature=0.1)
        
        self.small_calls += 1
            
        content = res.get("content", "")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        print(f'--- [Observation Compressed] ---\n{content}\n-------------------------------')
        return content

    def _call_llm(self, llm_client, messages: List[BaseMessage], stop: list = None) -> str:
        formatted = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, HumanMessage):
                role = "user"
            formatted.append({"role": role, "content": msg.content})

        res = llm_client.generate(formatted, temperature=0.1, stop=stop)
        content = clean_output(res.get("content", ""))
        
        if llm_client == self.plan_llm:
            self.large_calls += 1
            for k in self.total_usage:
                self.total_usage[k] += res.get("usage", {}).get(k, 0)
        else:
            self.small_calls += 1
            # We don't count small model tokens in total_usage as per user request
            
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content

    def _create_planner(self, state: Dict) -> Dict:
        tool_descriptions = "\\n".join(f"{i+1}. {tool.name}: {tool.description}" for i, tool in enumerate(self.tools))
        
        messages = state["messages"]
        replan = ""
        if len(messages) > 0 and isinstance(messages[-1], SystemMessage) and "Context from last attempt" in messages[-1].content:
            replan = (' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
                      "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
                      'You MUST use these information to create the next plan under "Current Plan".\\n'
                      ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\\n'
                      " - CRITICAL PLAN LIMITATION: You are an efficient AI assistant. You MUST ensure that the next plan contains NO MORE THAN 3 tasks (excluding the final join() call)!!!\\n"
                      " - MAXIMUM 3 TASKS PER PLAN - This is a HARD LIMIT. Do NOT exceed 3 tasks in the Current Plan.\\n"
                      " - Count your tasks: 1, 2, 3, then join(). NO MORE THAN 3 TASKS!!!\\n"
                      " - Focus on the most essential actions to provide concise, effective solutions.\\n"
                      " - REMINDER: Before creating your Current Plan, verify that it will have NO MORE THAN 3 tasks (plus join()).\\n"
                      " - CRITICAL: In the Current Plan, you MUST NEVER repeat the actions that are already executed in the Previous Plan. "
                      "All tasks from Previous Plan have already been executed and their results are available as Observations. \\n"
                      " - CRITICAL TASK NUMBERING: You MUST continue the task index from the end of the previous plan. "
                      "If the Previous Plan ended at task ID N, your Current Plan MUST start from task ID N+1. "
                      "DO NOT restart numbering from 1. DO NOT repeat task indices. "
                      "The task IDs must be continuous across the entire conversation. For example, if Previous Plan ended at task 2, "
                      "your Current Plan MUST start from task 3, then 4, 5, etc.\\n"
                      " - CRITICAL TOOL USAGE: You MUST ONLY use the tools that are provided in the tool list above. "
                      "DO NOT invent new tool names or use tools that are not listed. "
                      "Check the tool list carefully and use ONLY the exact tool names that are available. "
                      " - CRITICAL: Ensure every task in Current Plan follows the exact format: 'ID. tool_name(arguments)' with valid tool names and proper Python syntax.</no_think>")

            next_task = 0
            for message in messages[::-1]:
                if isinstance(message, FunctionMessage):
                    next_task = message.additional_kwargs.get("idx", 0) + 1
                    break
            if next_task > 0:
                messages[-1].content += f" - Begin counting at : {next_task}"
                
        messages_str = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if isinstance(msg, FunctionMessage):
                group = []
                while i < len(messages) and isinstance(messages[i], FunctionMessage):
                    m = messages[i]
                    idx = m.additional_kwargs.get("idx")
                    args = m.additional_kwargs.get("args", {})
                    group.append(f"Task {idx} [Tool: {m.name}] (Args: {args}):\n{m.content}")
                    i += 1
                
                group_str = "\n\n".join(group)
                if self.plan_llm != self.exec_llm:
                    group_str = self._compress_observation(group_str)
                messages_str.append(f"Observations:\n{group_str}")
            else:
                messages_str.append(f"{msg.type.capitalize()}: {msg.content}")
                i += 1

        prompt_text = self.PROMPT_TEMPLATE.format(
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
            replan=replan,
            messages="\n".join(messages_str)
        )
        
        plan_content = self._call_llm(self.plan_llm, [HumanMessage(content=prompt_text)], stop=["<END_OF_PLAN>"])
        if "join()" not in plan_content:
            next_task = 1
            for message in messages[::-1]:
                if isinstance(message, FunctionMessage):
                    next_task = max(next_task, message.additional_kwargs.get("idx", 0) + 1)
            plan_content += f"\n{next_task}. join()<END_OF_PLAN>"
            
        try:
            tasks = self.parser.parse(plan_content)
        except Exception as e:
            print(f"[Planner Error] Failed to parse plan content: {e}")
            tasks = []
            
        print(f"\n--- [Planner Output] Round {self._plan_round_counter} ---")
        print(plan_content)
        print("--------------------------\n")

        if self.plan_llm != self.exec_llm:
            # Transfer: plan sent from cloud planner to edge (Point 1)
            self.plan_l2s_bytes += len(str(plan_content).encode('utf-8'))

        return {"tasks_iter": tasks, "plan_content": plan_content}

    def _resolve_arg(self, arg: Union[str, Any], observations: Dict[int, Any]) -> Any:
        ID_PATTERN = r"\$\{?(\d+)\}?"
        def replace_match(match):
            idx = int(match.group(1))
            return str(observations.get(idx, match.group(0)))
            
        if isinstance(arg, str):
            return re.sub(ID_PATTERN, replace_match, arg)
        elif isinstance(arg, list):
            return [self._resolve_arg(a, observations) for a in arg]
        else:
            return arg

    def _execute_task(self, task: Task, observations: Dict[int, Any]) -> str:
        if task is None:
            return "ERROR(Task is None)"
            
        tool_to_use = task["tool"]
        if isinstance(tool_to_use, str):
            return tool_to_use
            
        args = task.get("args", {})
        resolved_args = {}
        try:
            if isinstance(args, str):
                resolved_args = self._resolve_arg(args, observations)
            elif isinstance(args, dict):
                resolved_args = {k: self._resolve_arg(v, observations) for k, v in args.items()}
            elif isinstance(args, list):
                schema = getattr(tool_to_use, "args_schema", None)
                field_names = list(schema.__fields__.keys()) if schema and hasattr(schema, "__fields__") else list(schema.model_fields.keys()) if schema else []
                if field_names:
                    resolved_args = {}
                    for i, val in enumerate(args):
                        if i < len(field_names):
                            resolved_args[field_names[i]] = self._resolve_arg(val, observations)
                else:
                    if len(args) == 1:
                        resolved_args = self._resolve_arg(args[0], observations)
                    else:
                        resolved_args = [self._resolve_arg(v, observations) for v in args]
            else:
                if getattr(tool_to_use, "name", "") == "get_current_date":
                    resolved_args = {}
                else:
                    resolved_args = args
                    
            if tool_to_use.name == "web_search" and isinstance(resolved_args, dict) and "query" in resolved_args:
                query_val = resolved_args["query"]
                if isinstance(query_val, str):
                    query_stripped = query_val.strip()
                    if query_stripped.startswith("{") and query_stripped.endswith("}"):
                        try:
                            parsed = json.loads(query_stripped)
                            if isinstance(parsed, dict) and "query" in parsed:
                                resolved_args["query"] = parsed["query"]
                            else:
                                resolved_args["query"] = query_stripped
                        except:
                            resolved_args["query"] = query_stripped
                    elif ", language=" in query_stripped:
                        parts = query_stripped.split(", language=", 1)
                        q_part = parts[0].strip().strip("'\"")
                        resolved_args["query"] = q_part
                        if "language" not in resolved_args:
                            resolved_args["language"] = parts[1].strip().rstrip(",)").strip("'\"")

        except Exception as e:
            return f"ERROR(Resolve args failed: {repr(e)})"
            
        action_name = tool_to_use.name
        action_input = resolved_args
        
        print(f"\n>>> [Tool Input] Task {task.get('idx', '?')} | Tool: {action_name}")
        print(f"Arguments: {action_input}")
            
        if action_name == "web_search":
            self.search_calls += 1
            
        input_tokens_est = 0
        if action_name == "code_generator":
            prompt_text = resolved_args.get("prompt", str(resolved_args)) if isinstance(resolved_args, dict) else str(resolved_args)
            input_tokens_est = len(str(prompt_text).split()) * 1.3
            

        try:
            if isinstance(resolved_args, dict):
                result = tool_to_use.invoke(resolved_args)
            else:
                if action_name == "web_search":
                    result = tool_to_use.invoke({"query": resolved_args})
                else:
                    result = tool_to_use.invoke(resolved_args)
                    
            # Track code tool input + output tokens (prefer API usage response, fallback to estimation)
            if action_name == "code_generator" and isinstance(result, dict):
                tokens_from_api = 0
                usage = result.get("usage") or result.get("token_usage")
                if isinstance(usage, dict):
                    if "total_tokens" in usage:
                        tokens_from_api = usage.get("total_tokens", 0)
                    else:
                        tokens_from_api = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

                if tokens_from_api > 0:
                    self.code_tokens += int(tokens_from_api)
                else:
                    code_text = result.get("generated_code", "")
                    output_tokens_est = len(code_text.split()) * 1.3
                    self.code_tokens += int(input_tokens_est + output_tokens_est)
                    
            if isinstance(result, dict):
                if action_name == "llm_inference":
                    usage = result.get("usage", {})
                    # Only count tokens if it's the cloud model (plan_llm)
                    if self.exec_llm == self.plan_llm:
                        self.large_calls += 1
                        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                            self.total_usage[k] += usage.get(k, 0)
                    else:
                        self.small_calls += 1

                if result.get("status") == "success":
                    final_res = str(result.get("result") or result.get("summary") or result.get("generated_code") or result.get("datetime") or result)
                else:
                    final_res = json.dumps(result, ensure_ascii=False)
            else:
                final_res = str(result)
            
            print(f"<<< [Tool Output] Task {task.get('idx', '?')} | Tool: {action_name} executed successfully.")
            print(f"Result: {final_res[:500]}{'...' if len(final_res) > 500 else ''}\n")
            return final_res
            
        except Exception as e:
            err_msg = f"ERROR(Execution failed: {repr(e)})"
            print(f"<<< [Tool Output] Task {task.get('idx', '?')} | Tool: {action_name} FAILED: {err_msg}\n")
            return err_msg

    def _execute_task_wrapper(self, task: Task, observations: Dict[int, Any]):
        if task is None: return
        try:
            obs = self._execute_task(task, observations)
        except Exception:
            obs = traceback.format_exc()
        observations[task["idx"]] = obs

    def schedule_pending_task(self, task: Task, observations: Dict[int, Any], retry_after: float = 0.2):
        if not task or "idx" not in task: return
        max_wait_time = 300.0
        max_iterations = int(max_wait_time / retry_after)
        
        for _ in range(max_iterations):
            deps = task.get("dependencies", [])
            if deps and any([dep not in observations for dep in deps]):
                time.sleep(retry_after)
                continue
            self._execute_task_wrapper(task, observations)
            return
            
        observations[task["idx"]] = f"ERROR(Task {task.get('idx')} timed out waiting for deps)"

    def plan_and_schedule(self, state: Dict) -> Dict:
        self._plan_round_counter += 1
        messages = state["messages"]
        observations = {}
        for msg in messages:
            if isinstance(msg, FunctionMessage):
                idx = msg.additional_kwargs.get("idx")
                if idx is not None:
                    observations[int(idx)] = msg.content
                    
        plan_result = self._create_planner(state)
        tasks_iter = plan_result["tasks_iter"]
        
        task_list = [t for t in tasks_iter if t is not None]
        
        futures = []
        new_observations = {}
        args_for_tasks = {}
        task_names = {}
        originals = set(observations.keys())
        
        with ThreadPoolExecutor() as executor:
            for task in task_list:
                if task is None: continue
                deps = task.get("dependencies", [])
                task_names[task["idx"]] = task["tool"] if isinstance(task["tool"], str) else task["tool"].name
                args_for_tasks[task["idx"]] = task.get("args", {})
                
                if deps and any([dep not in observations for dep in deps]):
                    futures.append(executor.submit(self.schedule_pending_task, task, observations, 0.25))
                else:
                    self._execute_task_wrapper(task, observations)
            wait(futures)
            
        tool_messages = []
        for k in sorted(observations.keys() - originals):
            obs = observations[k]
            tool_messages.append(
                FunctionMessage(
                    name=task_names.get(k, "unknown"),
                    content=str(obs),
                    additional_kwargs={"idx": k, "args": args_for_tasks.get(k, {})},
                    tool_call_id=str(k)
                )
            )
            
        return {
            "messages": tool_messages,
            "planned_tasks": task_list,
            "replan_count": self._replan_count
        }

    def joiner(self, state: Dict) -> Dict:
        messages = state["messages"]
        observations = []
        for msg in messages:
            if isinstance(msg, FunctionMessage):
                idx = msg.additional_kwargs.get("idx")
                if idx is not None:
                    if msg.content.strip().lower() == "join":
                        continue
                    args = msg.additional_kwargs.get("args", {})
                    observations.append(f"Task {idx} [Tool: {msg.name}] (Args: {args}):\n{msg.content}")
                    
        user_question = self._current_question
        
        focus_instruction = """
CRITICAL FOCUS REQUIREMENT - READ THIS CAREFULLY:
- Your PRIMARY goal is to answer the USER'S CURRENT QUESTION directly and accurately.
- All observations from previous tasks are ONLY SUPPORTING INFORMATION to help you answer the user's question.
- DO NOT start your response with meta-commentary like "Yes, the query has been answered" or "The observations show...".
- DO NOT mention Task IDs or tool names in your final response.
- PROVIDE THE ANSWER directly to the user as a helpful assistant would.
- ALWAYS focus on: "What is the user asking? How can I answer it directly?"
- If observations are incomplete or unclear, still try to provide the best answer to the user's question based on what you know.
- CRITICAL REPLANNING RULE: When you do not get the desired result from a tool, please check the tool's input ("Args"). If the tool's input is appropriate, then the issue is caused by the tool and is unrelated to the plan. In this case, you should directly return the answer to the user based on available information. Otherwise, if you need to obtain different information, perform replan again.
"""
        
        format_instruction = """
CRITICAL FORMAT REQUIREMENTS - YOU MUST USE THE FOLLOWING JSON FORMAT:
You MUST return a JSON object (and ONLY a JSON object) with TWO parameters:
1. "thought": A string containing your chain of thought reasoning
2. "action": An object containing exactly ONE of the following fields:
   - "response": Your final answer to the user's question, OR
   - "feedback": Feedback for replanning

Example FinalResponse:
{
  "thought": "I have all the information needed.",
  "action": {
    "response": "The answer is ..."
  }
}

Example Replan:
{
  "thought": "I need more information about X.",
  "action": {
    "feedback": "Please search for more details about X"
  }
}
"""
        
        obs_str = "\n\n".join(observations)
        if self.plan_llm != self.exec_llm and obs_str:
            obs_str = self._compress_observation(obs_str)
            # Transfer: execution information sent from edge to cloud joiner (Point 2)
            self.plan_s2l_bytes += len(obs_str.encode('utf-8'))
        
        joiner_prompt = f"""Based on the provided observations, generate the final answer to the user's query if possible, or decide if more planning is needed.
User query: {user_question}
Observations:
{obs_str}

{focus_instruction}

{format_instruction}
</no_think>
"""
        
        response_text = self._call_llm(self.plan_llm, [HumanMessage(content=joiner_prompt)])
        try:
            # First, try standard JSON loading
            match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            parsed = None
            if match:
                try:
                    # Try to handle common small model JSON errors by replacing literal newlines
                    json_str = match.group(1)
                    parsed = json.loads(json_str)
                except Exception:
                    # If still fails, try a more aggressive cleanup or regex extraction
                    pass
            
            # Robust extraction fallback: If JSON parsing fails, use regex to get fields directly
            # This is essential for small models that often forget to escape quotes or newlines
            if parsed is None:
                thought_match = re.search(r'"thought":\s*"(.*?)"', response_text, re.DOTALL)
                response_match = re.search(r'"response":\s*"(.*?)"', response_text, re.DOTALL)
                feedback_match = re.search(r'"feedback":\s*"(.*?)"', response_text, re.DOTALL)
                
                thought = thought_match.group(1) if thought_match else "Analyzing facts."
                if response_match:
                    parsed = {"thought": thought, "action": {"response": response_match.group(1)}}
                elif feedback_match:
                    parsed = {"thought": thought, "action": {"feedback": feedback_match.group(1)}}
                else:
                    raise ValueError("Could not extract thought/action from response via regex")

            thought = parsed.get("thought", "Analyzing the results.")
            action = parsed.get("action", {})
            response_msgs = [AIMessage(content=f"Thought: {thought}")]
            
            if "response" in action:
                return {"messages": response_msgs + [AIMessage(content=action["response"])]}
            elif "feedback" in action:
                return {"messages": response_msgs + [SystemMessage(content=f"Context from last attempt: {action['feedback']}")]}
            else:
                return {"messages": response_msgs + [SystemMessage(content="Context from last attempt: No valid action found. Replan.")]}
        except Exception as e:
            print(f"[Joiner Error] Critical parsing failure: {e}")
            return {
                "messages": [
                    AIMessage(content="Thought: Unable to parse joiner output, replanning is needed."),
                    SystemMessage(content=f"Context from last attempt: Failed to parse joiner decision ({str(e)}). Please create a new plan returning EXACTLY valid JSON.")
                ]
            }

    def _force_final_answer(self, state: Dict) -> Dict:
        messages = state["messages"]
        observations = []
        for msg in messages:
            if isinstance(msg, FunctionMessage):
                idx = msg.additional_kwargs.get("idx")
                if idx is not None:
                    if msg.content.strip().lower() == "join":
                        continue
                    args = msg.additional_kwargs.get("args", {})
                    observations.append(f"Task {idx} [Tool: {msg.name}] (Args: {args}):\n{msg.content}")
                    
        obs_str = "\n\n".join(observations)
        if self.plan_llm != self.exec_llm and obs_str:
            obs_str = self._compress_observation(obs_str)
                    
        user_question = self._current_question
        
        obs_str = f"The following are all the tasks that have been executed and their results:\n{obs_str}"
        if not observations:
            obs_str = "No task execution results are available."
            
        final_answer_prompt = f"""
CRITICAL FOCUS REQUIREMENT - READ THIS CAREFULLY:
- Your PRIMARY goal is to answer the USER'S CURRENT QUESTION directly and accurately.
- All observations from previous tasks are ONLY SUPPORTING INFORMATION to help you answer the user's question.
- DO NOT include meta-commentary like "Yes, the query has been answered" or "Based on the observations".
- DO NOT mention Task IDs or specific tools.
- Just PROVIDE THE ANSWER directly and clearly.
- ALWAYS focus on: "What is the user asking? How can I answer it directly?"
- If observations are incomplete or unclear, still try to provide the best answer to the user's question based on what you know.

USER'S CURRENT QUESTION (THIS IS YOUR PRIMARY FOCUS):
{user_question}

OBSERVATIONS FROM TASKS (THESE ARE ONLY SUPPORTING INFORMATION):
{obs_str}

Please generate a complete and accurate final answer to the user's question above. Focus on directly answering what the user asked. The observations are only there to help you - your main job is to answer the user's question clearly and directly.

Please return your answer directly without any additional formatting instructions or meta-talk.
</no_think>
"""
        response_text = self._call_llm(self.plan_llm, [HumanMessage(content=final_answer_prompt)])
        return {
            "messages": [
                AIMessage(content="Thought: Maximum replan limit reached, generating final answer based on all executed tasks' results."),
                AIMessage(content=response_text)
            ]
        }

    def _create_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("plan_and_schedule", self.plan_and_schedule)
        graph_builder.add_node("join", self.joiner)
        graph_builder.add_node("force_final_answer", self._force_final_answer)
        graph_builder.add_edge("plan_and_schedule", "join")
        
        def should_continue(state: Dict):
            messages = state["messages"]
            if isinstance(messages[-1], AIMessage):
                return END
                
            if self._replan_count >= self.max_iterations:
                print(f"\\n[Warning] Replan count exceeded max iterations ({self.max_iterations}). Forcing final answer.")
                return "force_final_answer"
                
            self._replan_count += 1
            print(f"\\n[Info] Replanning... (Count: {self._replan_count})")
            return "plan_and_schedule"
            
        graph_builder.add_conditional_edges("join", should_continue)
        graph_builder.add_edge("force_final_answer", END)
        graph_builder.add_edge(START, "plan_and_schedule")
        return graph_builder.compile()

    def run(self, user_query: str, history: List[Dict[str, str]] = []) -> Dict[str, Any]:
        self._current_question = user_query
        self._replan_count = 0
        self._plan_round_counter = 0
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.large_calls = 0
        self.small_calls = 0
        self.plan_s2l_bytes = 0
        self.plan_l2s_bytes = 0
        self.search_calls = 0
        self.code_tokens = 0
        
        # 端↔云通信统计（以UTF-8字节数精确记录）
        # 初始上传仅统计 query 和 history，系统提示词通常预置在云端。
        try:
            payload = {"user_query": user_query, "history": history}
            serialized = json.dumps(payload, ensure_ascii=False)
            self.plan_s2l_bytes += len(serialized.encode('utf-8'))
        except Exception:
            self.plan_s2l_bytes += len(str(user_query).encode('utf-8')) + len(str(history).encode('utf-8'))

        
        messages = []
        if history:
            # 添加历史对话记录的提示信息，帮助模型区分背景和当前问题
            messages.append(HumanMessage(content="The following are historical dialogue records provided to help you understand the context. Please focus on answering the current question presented at the end."))
            for msg in history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                else:
                    messages.append(AIMessage(content=msg.get("content", "")))
        messages.append(HumanMessage(content=f"Current Question to Answer: {user_query}"))



        final_state = self.chain.invoke({"messages": messages, "planned_tasks": [], "replan_count": 0})
        
        final_answer = "Failed to find answer."
        messages_out = final_state.get("messages", [])
        if messages_out and isinstance(messages_out[-1], AIMessage):
            final_answer = messages_out[-1].content
        
        # Transfer: final answer sent from cloud side to edge side (always occurs)
        self.plan_l2s_bytes += len(str(final_answer).encode('utf-8')) if final_answer else 0
            
        print(f"\\n=== [Final Answer] ===")
        print(final_answer)
        print("======================\\n")
            
        trace = []
        for msg in messages_out:
            if isinstance(msg, AIMessage) and str(msg.content).startswith("Thought:"):
                trace.append({"step": "thought", "content": msg.content})
            elif isinstance(msg, FunctionMessage):
                trace.append({"step": f"Task {msg.name}", "content": msg.content})
                
        return {
            "answer": final_answer,
            "trace": trace,
            "usage": self.total_usage,
            "total_calls": self.large_calls + self.small_calls,
            "small_calls": self.small_calls,
            "large_calls": self.large_calls,
            "transfer_stats": {"s2l": self.plan_s2l_bytes, "l2s": self.plan_l2s_bytes},
            "search_calls": getattr(self, "search_calls", 0),
            "code_tokens": getattr(self, "code_tokens", 0)
        }
