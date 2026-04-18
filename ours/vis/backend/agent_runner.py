import asyncio
import sys
import os
import time
import json
import logging
from io import StringIO
import contextlib
import re
# Import backend specific configurations and dependencies dynamically
AGENT_BASE_DIR = "/home/gujing/Graduation_Design/ours/vis/agent_core"
STATES_DIR = "/home/gujing/Graduation_Design/ours/vis/data/execution_states"

class WsLogHandler(logging.Handler):
    def __init__(self, broadcast_callback):
        super().__init__()
        self.broadcast = broadcast_callback
        self.loop = asyncio.get_event_loop()

    def emit(self, record):
        log_entry = self.format(record)
        # Safely broadcast from synchronous logging to async websocket
        asyncio.run_coroutine_threadsafe(
            self.broadcast({"type": "thought", "content": log_entry + "\n"}),
            self.loop
        )

class AppStdoutProxy(StringIO):
    def __init__(self, broadcast_callback):
        super().__init__()
        self.broadcast = broadcast_callback
        self.loop = asyncio.get_event_loop()

    def write(self, s):
        super().write(s)
        if s.strip():
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "thought", "content": s + "\n"}),
                self.loop
            )

class AgentRunner:
    def __init__(self, broadcast_callback):
        self.broadcast = broadcast_callback
        self.config_state = {
            "agent_type": "react",
            "memory_size": 5,
            "innovations": {"retrieval": True, "memory": True, "failure": True},
            "thresholds": {"t1": 0.9, "t2": 0.6}
        }
        self.memory_queue = []
        
        # Cumulative stats (Persistent across queries, cleared on new chat)
        self.total_llm_cost = 0.0
        self.total_search_count = 0
        self.total_code_cost = 0.0
        
        # Singleton refs
        self.large_llm = None
        self.small_llm = None
        self.exp_db = None
        self.agent_dir = ""
        
        # Map of msg_id -> execution_state for historical feedback (In-memory cache)
        self.execution_states = {}
        os.makedirs(STATES_DIR, exist_ok=True)
        self.last_execution_state = None # Keep for backward compatibility if needed

    async def update_config(self, new_config):
        self.config_state.update(new_config)
        # Prune memory if size decreased. 
        # Since memory now has roles, memory_size refers to "message pairs" (user + assistant)
        # So total messages = 2 * memory_size
        while len(self.memory_queue) > 2 * self.config_state["memory_size"]:
            self.memory_queue.pop(0)
        await self.broadcast({"type": "memory_queue", "items": self.memory_queue})

    async def clear_memory(self):
        self.memory_queue = []
        # Clear cumulative stats too
        self.total_llm_cost = 0.0
        self.total_search_count = 0
        self.total_code_cost = 0.0
        await self.broadcast({"type": "memory_queue", "items": self.memory_queue})
        await self.broadcast({
            "type": "cost", 
            "llm_cost": 0.0, 
            "total_llm_cost": 0.0,
            "search_count": 0,
            "total_search_count": 0,
            "code_cost": 0.0,
            "total_code_cost": 0.0,
            "time_spent": 0.0
        })

    async def load_conversation(self, data):
        """Restore backend state from a saved conversation"""
        self.memory_queue = data.get("memory", [])
        self.total_llm_cost = data.get("stats", {}).get("total_llm_cost", 0.0)
        self.total_search_count = data.get("stats", {}).get("total_search_count", 0)
        self.total_code_cost = data.get("stats", {}).get("total_code_cost", 0.0)
        
        # Restore agent_type
        if "agent_type" in data:
            self.config_state["agent_type"] = data["agent_type"]
            self._init_system(data["agent_type"])
        
        await self.broadcast({"type": "memory_queue", "items": self.memory_queue})
        await self.broadcast({
            "type": "cost",
            "llm_cost": 0.0,
            "total_llm_cost": self.total_llm_cost,
            "search_count": 0,
            "total_search_count": self.total_search_count,
            "code_cost": 0.0,
            "total_code_cost": self.total_code_cost,
            "time_spent": 0.0
        })

    def _init_system(self, agent_type):
        agent_dir = os.path.join(AGENT_BASE_DIR, "React" if agent_type == "react" else "LLMCompiler")
        self.agent_dir = agent_dir
        
        # Prevent module import clashing by clearing previous agent modules
        target_modules = ['agent', 'main', 'config', 'llm_client', 'experience_db', 'tool_db', 'tools', 'evaluation', 'dataset_utils', 'embedding_utils']
        for mod in list(sys.modules.keys()):
            # If the module name is in our target list or it was loaded from AGENT_BASE_DIR
            if mod in target_modules:
                m = sys.modules[mod]
                # Only clear if it was loaded from the agent_core directory to avoid breaking backend
                if hasattr(m, '__file__') and m.__file__ and AGENT_BASE_DIR in m.__file__:
                    del sys.modules[mod]
                
        # Prevent module import clashing
        if agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)
        elif sys.path[0] != agent_dir:
            # Ensure the current agent_dir is at the front
            sys.path.remove(agent_dir)
            sys.path.insert(0, agent_dir)
            
        os.chdir(agent_dir)
        import config
        from llm_client import LLMClient
        from experience_db import ExperienceDB
        
        # Override config based on UI
        config.SIMILARITY_THRESHOLD_1 = self.config_state["thresholds"]["t1"]
        config.SIMILARITY_THRESHOLD_2 = self.config_state["thresholds"]["t2"]
        config.TOOL_SIMILARITY_THRESHOLD = self.config_state["thresholds"].get("sim", 0.9)
        
        config.HIERARCHICAL_TRAJECTORY_RETRIEVAL = self.config_state["innovations"]["retrieval"]
        config.TOOL_MEMORY_LIBRARY = self.config_state["innovations"]["memory"]
        config.FAILURE_EXPERIENCE_UPDATE = self.config_state["innovations"]["failure"]
        
        self.large_llm = LLMClient(config.LARGE_MODEL_API_KEY, config.LARGE_MODEL_API_BASE, config.LARGE_MODEL_NAME)
        self.small_llm = LLMClient(config.small_model_api_key if hasattr(config, "small_model_api_key") else config.SMALL_MODEL_API_KEY, 
                                   config.small_model_api_base if hasattr(config, "small_model_api_base") else config.SMALL_MODEL_API_BASE, 
                                   config.small_model_name if hasattr(config, "small_model_name") else config.SMALL_MODEL_NAME)
        
        db_filename = "react_experience_db.json" if agent_type == "react" else "llm_compiler_experience_db.json"
        self.exp_db = ExperienceDB(f"/home/gujing/Graduation_Design/ours/vis/data/{db_filename}")

    async def process_query(self, query: str):
        # Generate unique ID for this execution
        import uuid
        msg_id = str(uuid.uuid4())
        await self.broadcast({"type": "msg_id", "msg_id": msg_id})
        await self.broadcast({"type": "thought", "content": "Agent is initiating workflow...\n"})

        # Add user query to memory
        self.memory_queue.append({"role": "user", "content": query})
        if len(self.memory_queue) > 2 * self.config_state["memory_size"]:
            self.memory_queue.pop(0)
        await self.broadcast({"type": "memory_queue", "items": self.memory_queue})

        ws_handler = WsLogHandler(self.broadcast)
        logging.getLogger().addHandler(ws_handler)
        logging.getLogger().setLevel(logging.INFO)
        old_stdout = sys.stdout
        sys.stdout = AppStdoutProxy(self.broadcast)
        
        start_time = time.time()
        final_answer = ""
        complex_mode = False
        mem_hit = False

        try:
            # We run the real process in a background thread
            # The current user query is already in memory_queue, so pass it as query
            # 1. Run real logic
            loop = asyncio.get_running_loop()
            def state_callback(service_loc, memory_hit=False):
                asyncio.run_coroutine_threadsafe(
                    self.broadcast({"type": "state", "complexity": service_loc, "memory_hit": memory_hit}),
                    loop
                )

            full_res, complex_mode, mem_hit, total_usage, latency, refined_query = await asyncio.to_thread(self._run_real_logic, query, state_callback)
            
            final_answer = full_res.get("answer", "")
            if not final_answer or final_answer == "No answer generated.":
                # Fallback: check if the last thought in trace contains a final answer
                trace = full_res.get("trace", [])
                if trace:
                    last_item = trace[-1]
                    if last_item.get("type") == "final":
                        final_answer = last_item.get("final_answer") or last_item.get("thought", "")
                    elif last_item.get("type") == "action":
                        final_answer = last_item.get("thought", "")
            
            if not final_answer:
                final_answer = "No answer generated."
            
            state = {
                "query": query,
                "refined_query": refined_query,
                "history": [m for m in self.memory_queue[:-1]],
                "full_res": full_res,
                "agent_type": self.config_state["agent_type"]
            }
            self.last_execution_state = state
            self.execution_states[msg_id] = state
            
            # Persist to disk for restart survival
            try:
                with open(os.path.join(STATES_DIR, f"{msg_id}.json"), 'w', encoding='utf-8') as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error persisting execution state: {e}")

            # Add assistant response to memory
            self.memory_queue.append({"role": "assistant", "content": final_answer})
            if len(self.memory_queue) > 2 * self.config_state["memory_size"]:
                self.memory_queue.pop(0)
            await self.broadcast({"type": "memory_queue", "items": self.memory_queue})

            # Metrics and stats broadcast moved here to avoid UnboundLocalError
            metrics = full_res.get("metrics", {})
            usage = metrics.get("token_usage", {})
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)
            llm_cost_estimate = 0.0
            if p_tok > 0 or c_tok > 0:
                llm_cost_estimate = (p_tok / 1000000.0) * 2.0 + (c_tok / 1000000.0) * 3.0
                self.total_llm_cost += llm_cost_estimate
                
            current_search_count = metrics.get("search_calls", 0)
            current_code_cost = 0.0
            # Realistic calculation for code generation tool: Input 2.25/MT, Output 9/MT
            code_in = metrics.get("code_input_tokens", 0)
            code_out = metrics.get("code_output_tokens", metrics.get("code_tokens", 0))
            if code_in > 0 or code_out > 0:
                # 2.25/1M for input, 9.0/1M for output
                current_code_cost = (code_in / 1000000.0) * 2.25 + (code_out / 1000000.0) * 9.0
            
            self.total_search_count += current_search_count
            self.total_code_cost += current_code_cost
            
            await self.broadcast({
                "type": "state", 
                "complexity": "Complex" if complex_mode else "Simple", 
                "memory_hit": mem_hit
            })
            
            await self.broadcast({
                "type": "cost", 
                "llm_cost": llm_cost_estimate, 
                "total_llm_cost": self.total_llm_cost,
                "search_count": current_search_count,
                "total_search_count": self.total_search_count,
                "code_cost": current_code_cost,
                "total_code_cost": self.total_code_cost,
                "time_spent": latency
            })
            
        except Exception as e:
            final_answer = f"Error during execution: {str(e)}"
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            logging.getLogger().removeHandler(ws_handler)

        await self.broadcast({"type": "response", "content": final_answer})
        await self.broadcast({"type": "done"})

    async def handle_feedback(self, status: str, msg_id: str = None):
        """Update Innovation 3 reflection if user is unsatisfied"""
        print(f"--- Handling Feedback: status={status}, msg_id={msg_id} ---")
        if status != "unsatisfied":
            return
            
        state = None
        if msg_id:
            if msg_id in self.execution_states:
                state = self.execution_states[msg_id]
            else:
                # Try loading from disk
                state_path = os.path.join(STATES_DIR, f"{msg_id}.json")
                if os.path.exists(state_path):
                    try:
                        with open(state_path, 'r', encoding='utf-8') as f:
                            state = json.load(f)
                            self.execution_states[msg_id] = state # Cache it
                    except Exception as e:
                        print(f"Error loading persisted state: {e}")
        
        if not state and self.last_execution_state:
            state = self.last_execution_state
            
        if not state:
            print(f"--- Feedback Error: No execution state found for ID {msg_id} ---")
            return
            
        # Ensure exp_db is initialized for the agent type in the state
        agent_type_in_state = state.get("agent_type", self.config_state["agent_type"])
        if not self.exp_db or self.config_state["agent_type"] != agent_type_in_state:
            self.config_state["agent_type"] = agent_type_in_state
            self._init_system(agent_type_in_state)
            
        if not self.exp_db:
             print(f"--- Feedback Error: Failed to initialize experience DB for {agent_type_in_state} ---")
             return
        full_res = state["full_res"]
        
        # Log the conditions
        print(f"--- Feedback Target: Model={full_res.get('model')}, Category={full_res.get('category')} ---")
        
        # Only reflect on small model tasks using experience (Innovation 3)
        # We only reflect if it was a complex task (category 0) AND memory hit (using hierarchical experience)
        method = full_res.get("method", "")
        # "hierarchical" for LLMCompiler, "hybrid_small" for React
        is_mem_hit = "hierarchical" in method or "hybrid_small" in method
        if full_res.get("category") == 0 and is_mem_hit:
            print(f"--- Triggering Manual Reflection for {state['agent_type']} ---")
            
            # Reuse the legacy logic from experience_db.py but with manual trigger
            from experience_db import reflect_on_failure
            
            refined_query = state["refined_query"]
            ref_exp = full_res.get("used_experience") or full_res.get("exp_param", "")
            trace = full_res.get("trace", [])
            history = state["history"]
            
            # We treat manual dissatisfaction as score 0
            reflection, usage = await asyncio.to_thread(reflect_on_failure, self.large_llm, refined_query, ref_exp, trace, 0.0, history)
            
            if reflection:
                print(f"--- Saving Manual Reflection ---")
                self.exp_db.update_reflection(full_res.get("exp_data", {}).get("query") or refined_query, reflection)
                
                # Update cost with reflection cost
                p_tok = usage.get("prompt_tokens", 0)
                c_tok = usage.get("completion_tokens", 0)
                cost = (p_tok / 1000000.0) * 2.0 + (c_tok / 1000000.0) * 3.0
                self.total_llm_cost += cost
                
                await self.broadcast({
                    "type": "cost",
                    "llm_cost": cost,
                    "total_llm_cost": self.total_llm_cost,
                    "total_search_count": self.total_search_count,
                    "total_code_cost": self.total_code_cost,
                    "time_spent": 0.0
                })
                await self.broadcast({"type": "thought", "content": f"Innovation 3 Active: Added manual reflection to experience DB.\n"})
                await self.broadcast({"type": "done"})

    def _run_real_logic(self, query, state_callback=None):
        agent_type = self.config_state["agent_type"]
        self._init_system(agent_type)
        
        # history is everything in memory_queue EXCEPT the last user query
        history_formatted = [m for m in self.memory_queue[:-1]]
        
        # Dynamic import from the current agent directory
        import sys
        if '' not in sys.path: sys.path.insert(0, '')
        import main
        import importlib
        importlib.reload(main)
        
        # Dispatch to core logic
        if agent_type == "react":
            from agent import ReactAgent
            from tool_db import ToolDB
            tool_db_edge = ToolDB("/home/gujing/Graduation_Design/ours/vis/data/edge_tool_db.json")
            tool_db_cloud = ToolDB("/home/gujing/Graduation_Design/ours/vis/data/cloud_tool_db.json")
            agent_small_inst = ReactAgent(self.small_llm, is_edge=True, edge_tool_db=tool_db_edge, cloud_tool_db=tool_db_cloud)
            agent_large_inst = ReactAgent(self.large_llm, is_edge=False, edge_tool_db=tool_db_edge, cloud_tool_db=tool_db_cloud, generate_experience=True)
            
            full_res = main.run_single_query(
                user_query=query,
                history=history_formatted,
                large_llm=self.large_llm,
                small_llm=self.small_llm,
                agent_large=agent_large_inst,
                agent_small=agent_small_inst,
                exp_db=self.exp_db,
                evaluator=None, # No auto evaluation in visualizer
                state_callback=state_callback
            )
        else: # llmcompiler
            from agent import LLMCompileAgent
            from tool_db import ToolMemoryDB
            tool_db_edge = ToolMemoryDB("/home/gujing/Graduation_Design/ours/vis/data/edge_tool_db.json")
            tool_db_cloud = ToolMemoryDB("/home/gujing/Graduation_Design/ours/vis/data/cloud_tool_db.json")
            
            # Note: LLMCompiler's run_single_query has 'db' as argument name
            full_res = main.run_single_query(
                user_query=query,
                history=history_formatted,
                large_llm=self.large_llm,
                small_llm=self.small_llm,
                db=self.exp_db,
                evaluator=None,
                state_callback=state_callback
            )

        complex_mode = full_res.get("category") == 0
        method = full_res.get("method", "")
        mem_hit = "hierarchical" in method or "hybrid_small" in method
        # latency and refined_query from metrics
        m = full_res.get("metrics", {})
        total_usage = m.get("token_usage", {})
        latency = m.get("latency_seconds", 0.0)
        refined_query = full_res.get("refined_query") or full_res.get("question") or query
        
        return full_res, complex_mode, mem_hit, total_usage, latency, refined_query
