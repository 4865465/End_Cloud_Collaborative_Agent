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
        self.records = []
        self.index = None
        self.d = 1024  # QwenEmbedding default dimension
        self.load_db()
        
    def load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.records = json.load(f)
            except Exception as e:
                print(f"Error loading tool memory db JSON: {e}")
                self.records = []
        
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                print(f"Error loading FAISS index for ToolMemory: {e}")
                self.index = faiss.IndexFlatIP(self.d)
        else:
            self.index = faiss.IndexFlatIP(self.d)

    def save_db(self):
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
            faiss.write_index(self.index, self.index_path)
        except Exception as e:
            print(f"Error saving tool memory db: {e}")
            
    def add_record(self, tool_name: str, tool_input: str, tool_output: str) -> None:
        emb = self.encoder.encode_single(f"{tool_name}: {tool_input}")
        emb_np = np.array([emb]).astype('float32')
        faiss.normalize_L2(emb_np)
        self.index.add(emb_np)
        
        self.records.append({
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output
        })
        self.save_db()

    def search(self, tool_name: str, tool_input: str, threshold: float) -> Tuple[bool, str]:
        if not self.records or self.index.ntotal == 0:
            return False, ""

        query_text = f"{tool_name}: {tool_input}"
        query_emb = self.encoder.encode_single(query_text)
        query_np = np.array([query_emb]).astype('float32')
        faiss.normalize_L2(query_np)

        scores, indices = self.index.search(query_np, 5) # Search top 5 to find same tool
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0: continue
            best_record = self.records[idx]
            if best_record["tool_name"] == tool_name and score > threshold:
                return True, best_record["tool_output"]
            
        return False, ""
