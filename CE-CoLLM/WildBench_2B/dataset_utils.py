"""
Data Loading Utilities
"""
from datasets import load_dataset
from typing import List, Dict, Any
import os
import json
import time
from config import DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT, HF_TOKEN, DATASET_LIMIT


def load_wildbench(sample_size: int = DATASET_LIMIT) -> List[Dict[str, Any]]:
    """
    Load WildBench dataset and format for evaluation.
    Each item is converted directly into a task dictionary.
    """
    try:
        token = os.getenv("HF_TOKEN", HF_TOKEN)

        if not token:
            print("Warning: HuggingFace Token not found. Set HF_TOKEN environment variable.")

        max_retries = 5
        base_delay = 15

        dataset = None
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                print(f"Loading dataset {DATASET_NAME} with config {DATASET_CONFIG}... (attempt {attempt + 1}/{max_retries})")
                dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, token=token)
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)
                    print(
                        f"Error loading dataset from HuggingFace, retry in {wait:.1f}s "
                        f"({attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(wait)
                else:
                    print(f"Error loading dataset after {max_retries} attempts: {e}")

        if dataset is None:
            raise last_error if last_error is not None else RuntimeError("Unknown error loading dataset")

        if sample_size is not None:
            limit = min(sample_size, len(dataset))
            dataset = dataset.select(range(limit))
            print(f"Sampling {limit} conversations from {DATASET_NAME}.")
        
        tasks = []
        for i, item in enumerate(dataset):
            session_id = item.get("session_id")
            item_id = item.get("id")
            task_id = str(session_id) if session_id else (str(item_id) if item_id else f"task_{i}")
            
            # Clean conversation_input: keep only role and content
            raw_conv = item.get("conversation_input", [])
            clean_conv = []
            for msg in raw_conv:
                clean_conv.append({
                    "role": msg.get("role"),
                    "content": msg.get("content")
                })
            
            if not clean_conv:
                continue
                
            # The last user message is the input, others are history
            # WildBench usually ends with a user message
            last_msg = clean_conv[-1]
            history = clean_conv[:-1]
            
            tasks.append(
                {
                    "task_id": task_id,
                    "session_id": session_id,
                    "id": item_id,
                    "history": history,
                    "input": last_msg.get("content", ""),
                    "references": item.get("references", {}),
                    "checklist": item.get("checklist", []),
                }
            )

        print(f"Successfully loaded {len(tasks)} WildBench tasks.")
        return tasks

    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Return mock data for development if real data fails completely
        return [
            {
                "task_id": f"mock_{i}",
                "session_id": f"sess_{i}",
                "input": f"Mock Question {i}: What is the capital of France?",
                "history": [],
                "references": {"mock": "Paris"},
                "checklist": ["Helpful", "Correct"],
            }
            for i in range(5)
        ]
