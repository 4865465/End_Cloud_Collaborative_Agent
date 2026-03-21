import json
import re
import os
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from embedding_utils import QwenEmbedding

class ExperienceDB:
    def __init__(self, db_path: str = "experience_db.json"):
        self.db_path = db_path
        self.index_path = db_path.replace(".json", ".index")
        self.encoder = QwenEmbedding()
        self.records = []
        self.index = None
        self.d = 1024  # QwenEmbedding default dimension
        self.load_db()
        
    def load_db(self):
        # 1. Load Metadata
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.records = json.load(f)
            except Exception as e:
                print(f"Error loading experience db JSON: {e}")
                self.records = []
        
        # 2. Load/Init Index
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self.index = faiss.IndexFlatIP(self.d)
        else:
            self.index = faiss.IndexFlatIP(self.d)

        # 3. Migration
        has_embeddings_in_json = any("embedding" in r for r in self.records)
        if has_embeddings_in_json:
            print(f"Migrating embeddings from {self.db_path} to FAISS index...")
            all_embs = []
            for r in self.records:
                if "embedding" in r:
                    all_embs.append(r.pop("embedding"))
                else:
                    all_embs.append(self.encoder.encode_single(r.get("query", "")))
            
            if all_embs:
                embs_np = np.array(all_embs).astype('float32')
                faiss.normalize_L2(embs_np)
                self.index = faiss.IndexFlatIP(self.d)
                self.index.add(embs_np)
                self.save_db()

        # Sync check
        if self.index.ntotal != len(self.records) and len(self.records) > 0:
            print(f"Warning: Index count ({self.index.ntotal}) mismatch with records count ({len(self.records)}). Rebuilding index...")
            self.rebuild_index()

    def rebuild_index(self):
        print("Re-encoding all experience records...")
        all_embs = []
        for r in self.records:
            all_embs.append(self.encoder.encode_single(r.get("query", "")))
        if all_embs:
            embs_np = np.array(all_embs).astype('float32')
            faiss.normalize_L2(embs_np)
            self.index = faiss.IndexFlatIP(self.d)
            self.index.add(embs_np)
            self.save_db()
                
    def save_db(self):
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
            faiss.write_index(self.index, self.index_path)
        except Exception as e:
            print(f"Error saving experience db: {e}")
            
    def add_experience(self, query: str, trace: Any, experience: str) -> None:
        # Encode
        emb = self.encoder.encode_single(query)
        emb_np = np.array([emb]).astype('float32')
        faiss.normalize_L2(emb_np)
        self.index.add(emb_np)
        
        # Add metadata
        self.records.append({
            "query": query,
            "trace": trace,
            "experience": experience,
            "reflection": "",
        })
        self.save_db()
        

    def search_by_emb(self, query_emb: np.ndarray, threshold1: float) -> Tuple[bool, Dict[str, Any]]:
        if not self.records or self.index.ntotal == 0:
            return False, {}

        # Normalize query
        query_np = query_emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_np)

        # Search top 1
        scores, indices = self.index.search(query_np, 1)
        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])
        
        if best_idx < 0:
            return False, {}
            
        best_record = self.records[best_idx]
        if best_score > threshold1:
            return True, {
                "action": "experience",
                "score": best_score,
                "query": best_record.get("query", ""),
                "trace": best_record.get("trace", []),
                "experience": best_record.get("experience", ""),
                "reflection": best_record.get("reflection", "")
            }
            
        return False, {}

    def search(self, query: str, threshold1: float) -> Tuple[bool, Dict[str, Any]]:
        query_emb = self.encoder.encode_single(query)
        return self.search_by_emb(query_emb, threshold1)



def generate_experience_summary(llm, user_query: str, trace: List[Dict[str, Any]], history: List[Dict[str, str]] = []) -> Tuple[str, Dict[str, Any]]:
    """
    根据 trace 总结经验，仅在评估分高的情况下调用。
    """
    trace_str = json.dumps(trace, ensure_ascii=False, indent=2)
    history_str = ""
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
    else:
        history_str = "None"

    system_prompt = (
        "You are a workflow analysis engine. Your ONLY task is to generate a single-sentence experience summary. "
        "DO NOT provide any analysis, introduction, or reasoning in your response. "
        "Output ONLY the finalized formula string."
    )

    exp_instr = (
        "### OUTPUT FORMAT RULE ###\n"
        "You must return ONLY a single line starting with 'Experience Summary:'.\n"
        "Formula: Experience Summary: If the user purpose is [Intent], then the workflow should be [Specific Tool Sequence & Dependencies].\n"
        "\n"
        "GUIDELINE FOR [Specific Tool Sequence & Dependencies]:\n"
        "Describe the parallelizable DAG-style plan. e.g., 'Simultaneously run web_search for A and B, then use llm_inference on $1 and $2 to finalize'.\n"
        "CRITICAL: Direct answer only. No explanations."
    )

    user_prompt = (
        "Analyze the execution trace and provide the best-practice logic.\n\n"
        f"[Conversation Context]:\n{history_str}\n"
        f"[User Query]: {user_query}\n"
        f"[Execution Trace]:\n{trace_str}\n\n"
        "---"
        f"{exp_instr}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        # Use temperature 0.0 for maximum adherence
        res = llm.generate(messages, max_tokens=500, temperature=0.1)
        text = res.get("content", "").strip()
        # Remove reasoning tags if any
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        usage = res.get("usage", usage)
        
        # Robust extraction
        # 1. Priority: If "Experience Summary:" is present, take everything after it
        match = re.search(r"Experience Summary:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        else:
            text = text.strip()
        # Ensure consistent punctuation
        if text and not text.endswith("."):
            text += "."
        
        return text, usage
    except Exception as e:
        print(f"Failed to generate experience summary: {e}")
        return "", usage


def analyze_task(llm, query: str, history: List[Dict[str, str]]) -> Tuple[str, int, Dict[str, Any]]:
    """
    分析任务：整理意图并初步判断复杂度。
    1: 简单任务（闲聊、极其简单的事实），小模型可直接回答。
    0: 复杂任务（需要工具、搜索、计算或复杂逻辑），进入后续混合策略（检索工具/经验）。
    """
    # Format history
    history_str = ""
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
    else:
        history_str = "None"
    
    prompt = (
        "You are an expert AI task analyzer in the LLMCompiler framework.\n\n"
        "Step 1: Refine the user's intent. Incorporate context from history into a standalone query.\n"
        "Step 2: Decide the execution path.\n"
        "  - Category 1 (Direct Path): Greetings, simple facts, or single-step logic that doesn't need planning or external search. The small model can answer this immediately.\n"
        "  - Category 0 (Compiler Path): Complex queries involving multi-step reasoning, calculations, web search, or code generation. These require the LLMCompiler parallel planning and tool execution engine.\n"
        "\n"
        "CRITICAL: When in doubt, choose Category 0. We prefer the robust compiler pipeline for anything requiring non-trivial tool utilization or depth.\n\n"
        f"[History]:\n{history_str}\n"
        f"[Current Query]: {query}\n\n"
        "Output Format:\n"
        "Intent: <refined_intent>\n"
        "Category: <0 or 1>\n"
        "</no_think>"
    )
    
    try:
        res = llm.generate([{"role": "user", "content": prompt}], max_tokens=500)
        text = res.get("content", "")
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # 解析输出
        refined_query = query
        category = 0 # 默认保守
        
        intent_match = re.search(r"Intent:\s*(.*)", text, re.IGNORECASE)
        if intent_match:
            refined_query = intent_match.group(1).strip()
        
        category_match = re.search(r"Category:\s*([01])", text, re.IGNORECASE)
        if category_match:
            category = int(category_match.group(1).strip())
            
        usage = res.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        return refined_query, category, usage
    except Exception as e:
        print(f"Failed to analyze task: {e}")
        return query, 0, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}