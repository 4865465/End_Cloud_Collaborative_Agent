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
        
    def update_reflection(self, query: str, reflection: str) -> None:
        for rec in self.records:
            if rec["query"] == query:
                if rec.get("reflection"):
                    rec["reflection"] += "\n" + reflection
                else:
                    rec["reflection"] = reflection
                self.save_db() # Refresh JSON
                break

    def search_by_emb(self, query_emb: np.ndarray, threshold1: float, threshold2: float) -> Tuple[bool, Dict[str, Any]]:
        from config import HIERARCHICAL_TRAJECTORY_RETRIEVAL
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
                
        if HIERARCHICAL_TRAJECTORY_RETRIEVAL:
            if best_score > threshold1:
                return True, {
                    "action": "trace",
                    "score": best_score,
                    "query": best_record.get("query", ""),
                    "trace": best_record.get("trace", []),
                    "experience": best_record.get("experience", ""),
                    "reflection": best_record.get("reflection", "")
                }
            elif best_score > threshold2:
                return True, {
                    "action": "experience",
                    "score": best_score,
                    "query": best_record.get("query", ""),
                    "trace": best_record.get("trace", []),
                    "experience": best_record.get("experience", ""),
                    "reflection": best_record.get("reflection", "")
                }
        else:
            if best_score > threshold2:
                return True, {
                    "action": "experience",
                    "score": best_score,
                    "query": best_record.get("query", ""),
                    "trace": best_record.get("trace", []),
                    "experience": best_record.get("experience", ""),
                    "reflection": best_record.get("reflection", "")
                }
            
        return False, {}

    def search(self, query: str, threshold1: float, threshold2: float) -> Tuple[bool, Dict[str, Any]]:
        query_emb = self.encoder.encode_single(query)
        return self.search_by_emb(query_emb, threshold1, threshold2)






def reflect_on_failure(large_llm, query: str, ref_exp: str, trace: Any, score: float, history: List[Dict[str, str]] = []) -> Tuple[str, Dict[str, Any]]:
    trace_str = json.dumps(trace, ensure_ascii=False, indent=2)
    history_str = ""
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
    else:
        history_str = "None"
        
    prompt = (
        f"You are an expert AI agent. A small model was tasked to solve a user query.\n"
        f"It was provided with a reference past experience/trace, but it still failed and achieved a low score of {score}.\n"
        f"[Conversation History]:\n{history_str}\n"
        f"[User Query]: {query}\n"
        f"[Reference Experience provided to small model]: {ref_exp}\n"
        f"[Small Model's Execution Trace]:\n{trace_str}\n\n"
        f"Your task is to analyze why the small model failed despite the reference information. "
        f"Provide a concise, highly actionable 'Reflection' addressing the gap or common pitfall "
        f"so that future executions will avoid this mistake.\n\n"
        f"Reflection:</no_think>"
    )
    try:
        messages = [{"role": "user", "content": prompt}]
        res = large_llm.generate(messages, max_tokens=600)
        text = res.get("content", "")
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        usage = res.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        return text, usage
    except Exception as e:
        print(f"Failed to generate reflection: {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}