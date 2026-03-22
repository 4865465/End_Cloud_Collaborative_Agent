import json
import os
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from embedding_utils import QwenEmbedding

class ToolMemoryDB:
    def __init__(self, db_path: str = "tool_memory_db.json"):
        self.db_path = db_path
        self.index_path = db_path.replace(".json", ".index")
        self.encoder = QwenEmbedding()
        self.records = {}  # Grouped by tool_name: Dict[str, List[Dict]]
        self.index = None
        self.d = 1024  # QwenEmbedding default dimension
        self.load_db()
        
    def load_db(self):
        # 1. Load Metadata
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Migration: Convert list to dict grouped by tool_name
                        self.records = {}
                        for i, r in enumerate(data):
                            t_name = r.get("tool_name", "unknown")
                            if t_name not in self.records:
                                self.records[t_name] = []
                            if "global_idx" not in r:
                                r["global_idx"] = i
                            self.records[t_name].append(r)
                    else:
                        self.records = data
            except Exception as e:
                print(f"Error loading tool memory db JSON: {e}")
                self.records = {}
        
        # 2. Handle FAISS Index
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                print(f"Error loading FAISS index for ToolMemory: {e}")
                self.index = faiss.IndexFlatIP(self.d)
        else:
            self.index = faiss.IndexFlatIP(self.d)

        # 3. Sync check / Rebuild if needed
        all_recs = []
        for recs in self.records.values():
            all_recs.extend(recs)
        
        if self.index.ntotal != len(all_recs) and len(all_recs) > 0:
            print(f"Warning: ToolMemory index count ({self.index.ntotal}) mismatch with records ({len(all_recs)}). Rebuilding...")
            self.rebuild_index()

    def rebuild_index(self):
        all_recs = []
        for recs in self.records.values():
            all_recs.extend(recs)
        all_recs.sort(key=lambda x: x.get("global_idx", 0))

        all_embs = []
        for i, r in enumerate(all_recs):
            # Recalculate embedding if necessary
            txt = r.get("tool_input", "")
            all_embs.append(self.encoder.encode_single(txt))
            r["global_idx"] = i # Ensure sync
        
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
            print(f"Error saving tool memory db: {e}")
            
    def add_record(self, tool_name: str, tool_input: str, tool_output: str) -> None:
        emb = self.encoder.encode_single(tool_input)
        emb_np = np.array([emb]).astype('float32')
        faiss.normalize_L2(emb_np)
        
        global_idx = self.index.ntotal
        self.index.add(emb_np)
        
        if tool_name not in self.records:
            self.records[tool_name] = []
            
        self.records[tool_name].append({
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": str(tool_output),
            "global_idx": global_idx
        })
        self.save_db()

    def search(self, tool_name: str, tool_input: str, threshold: float) -> Tuple[bool, str]:
        if not self.records or self.index.ntotal == 0:
            return False, ""

        tool_records = self.records.get(tool_name, [])
        if not tool_records:
            return False, ""

        # Encode query
        query_emb = self.encoder.encode_single(tool_input)
        query_np = np.array([query_emb]).astype('float32')
        faiss.normalize_L2(query_np)

        # Reconstruct vectors for this specific tool and calc inner product
        # (This avoids global filtering and is more precise)
        try:
            indices = [r["global_idx"] for r in tool_records]
            tool_embs = np.array([self.index.reconstruct(int(i)) for i in indices]).astype('float32')
            
            # Since vectors and query are normalized, dot product is cosine similarity
            scores = np.dot(tool_embs, query_np.T).flatten()
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            
            if best_score > threshold:
                return True, tool_records[best_idx]["tool_output"]
        except Exception as e:
            print(f"Error during tool reconstruction search: {e}")
            
        return False, ""
