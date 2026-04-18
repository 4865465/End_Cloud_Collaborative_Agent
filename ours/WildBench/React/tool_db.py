import json
import os
import sys
import numpy as np
import faiss
from typing import Tuple

sys.path.append("/home/gujing/LMSYS_CHAT_1M")
from embedding_utils import QwenEmbedding

class ToolDB:
    def __init__(self, db_path: str = "tool_db.json"):
        self.db_path = db_path
        self.index_path = db_path.replace(".json", ".index")
        self.encoder = QwenEmbedding()
        self.records = {}  # Changed to Dict[str, List[Dict]]
        self.index = None
        self.d = 1024  # QwenEmbedding default dimension
        self.load_db()

    def load_db(self):
        # 1. Load Metadata from JSON
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Migration: convert list to dict grouped by tool_name
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
                print(f"Error loading tool db JSON: {e}")
                self.records = {}
        
        # 2. Handle FAISS Index
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self.index = faiss.IndexFlatIP(self.d)
        else:
            self.index = faiss.IndexFlatIP(self.d)
            
        # 3. Migration / Sync check
        all_records = []
        for recs in self.records.values():
            all_records.extend(recs)
        all_records.sort(key=lambda x: x.get("global_idx", 0))

        has_embeddings_in_json = any("embedding" in r for r in all_records)
        if has_embeddings_in_json:
            print(f"Migrating embeddings from {self.db_path} to FAISS index...")
            all_embs = []
            for r in all_records:
                if "embedding" in r:
                    all_embs.append(r.pop("embedding"))
                else:
                    txt = r.get("tool_input", "")
                    all_embs.append(self.encoder.encode_single(txt))
            
            if all_embs:
                embs_np = np.array(all_embs).astype('float32')
                faiss.normalize_L2(embs_np)
                self.index = faiss.IndexFlatIP(self.d)
                self.index.add(embs_np)
                # Re-assign global_idx after migration to match FAISS index
                for i, r in enumerate(all_records):
                    r["global_idx"] = i
                self.save_db()
        
        if self.index.ntotal != len(all_records) and len(all_records) > 0:
            print(f"Warning: Index count ({self.index.ntotal}) mismatch with records count ({len(all_records)}). Rebuilding index...")
            self.rebuild_index()

    def rebuild_index(self):
        print("Re-encoding all tool records...")
        all_recs = []
        for recs in self.records.values():
            all_recs.extend(recs)
        all_recs.sort(key=lambda x: x.get("global_idx", 0))

        all_embs = []
        for i, r in enumerate(all_recs):
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
            # Save metadata only (no embeddings)
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
            # Save binary index
            faiss.write_index(self.index, self.index_path)
            # print(f"Saved DB and Index to {self.db_path}")
        except Exception as e:
            print(f"Error saving tool db: {e}")

    def add_record(self, tool_name: str, tool_input: str, tool_output: str) -> None:
        if tool_input is None:
            tool_input = ""
        
        # 1. Encode and add to FAISS
        emb = self.encoder.encode_single(tool_input)
        emb_np = np.array([emb]).astype('float32')
        faiss.normalize_L2(emb_np)
        idx = self.index.ntotal
        self.index.add(emb_np)
        
        # 2. Add to metadata
        if tool_name not in self.records:
            self.records[tool_name] = []
            
        self.records[tool_name].append({
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": str(tool_output),
            "global_idx": idx
        })
        self.save_db()

    def search_by_emb(self, tool_name: str, query_emb: np.ndarray, threshold: float) -> Tuple[bool, str]:
        if not self.records or self.index.ntotal == 0:
            return False, ""
        
        # Look up records for the specific tool directly
        tool_records = self.records.get(tool_name, [])
        if not tool_records:
            return False, ""
        
        ids = [r["global_idx"] for r in tool_records]

        # Normalize query
        query_np = query_emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_np)

        # For a small number of records, we can reconstruct and calculate similarity.
        # This is more efficient than searching a full index and then filtering if the filter is very selective.
        # Plus, reconstruct avoids storing embeddings in metadata.
        
        # Reconstruct vectors for the specific tool
        tool_embs = np.array([self.index.reconstruct(int(i)) for i in ids]).astype('float32')
        
        # Calculate Inner Product (already normalized, so this is cosine similarity)
        scores = np.dot(tool_embs, query_np.T).flatten()
        best_idx_in_tool = np.argmax(scores)
        best_score = scores[best_idx_in_tool]
        
        if best_score > threshold:
            return True, tool_records[best_idx_in_tool]["tool_output"]
        
        return False, ""

    def search(self, tool_name: str, tool_input: str, threshold: float) -> Tuple[bool, str]:
        if not tool_input:
            tool_input = ""
        query_emb = self.encoder.encode_single(tool_input)
        return self.search_by_emb(tool_name, query_emb, threshold)

