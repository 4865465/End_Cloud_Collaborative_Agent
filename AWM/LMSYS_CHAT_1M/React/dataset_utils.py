"""
Data Loading Utilities
"""
from datasets import load_dataset
from typing import List, Dict, Any
import os
import json
import time
from config import DATASET_NAME, DATASET_SPLIT, SAMPLE_SIZE_Multi, SAMPLE_SIZE_Single, HF_TOKEN


def load_lmsys_chat_1m(sample_size: int = None) -> List[Dict[str, Any]]:
    """
    Load LMSYS CHAT 1M dataset and filter for English conversations.

    为远程数据集加载增加重试机制，防止临时网络/服务问题导致整体流程失败。
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
                print(f"Loading dataset with token check... (attempt {attempt + 1}/{max_retries})")
                dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, token=token)
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

        dataset = dataset.select(range(min(10000, len(dataset))))
        
        print(f"Filtering English conversations...")
        english_dataset = dataset.filter(lambda x: x.get("language") == "English")
        
        if sample_size is not None:
            # If sample_size is explicitly provided, follow it as a total limit
            limit = min(sample_size, len(english_dataset))
            print(f"Sampling {limit} conversations from filtered English dataset (override).")
            selected_dataset = english_dataset.select(range(limit))
        else:
            # Separate sampling for Multi-turn and Single-turn
            print(f"Performing split sampling: {SAMPLE_SIZE_Multi} multi and {SAMPLE_SIZE_Single} single turns.")
            
            # Multi-turn sampling
            multi_dataset = english_dataset.filter(lambda x: x.get("turn") > 1)
            multi_limit = min(SAMPLE_SIZE_Multi, len(multi_dataset))
            multi_samples = multi_dataset.select(range(multi_limit))
            print(f"Sampled {multi_limit} multi-turn conversations.")
            
            # Single-turn sampling
            single_dataset = english_dataset.filter(lambda x: x.get("turn") == 1)
            single_limit = min(SAMPLE_SIZE_Single, len(single_dataset))
            single_samples = single_dataset.select(range(single_limit))
            print(f"Sampled {single_limit} single-turn conversations.")
            
            from datasets import concatenate_datasets
            selected_dataset = concatenate_datasets([multi_samples, single_samples])

        conversations = []
        for i, item in enumerate(selected_dataset):
            conv_id = item.get("conversation_id") or item.get("id") or f"conv_{i}"

            # 提取对话内容
            if "conversation" in item:
                full_conv = item["conversation"]
            elif "messages" in item:
                full_conv = item["messages"]
            else:
                full_conv = item

            conversations.append(
                {
                    "conversation_id": conv_id,
                    "conversation": full_conv,
                    "language": item.get("language"),  # 可选：保留语言标签以便核对
                    "original_idx": i,
                }
            )

        print(f"Successfully loaded {len(conversations)} English conversations.")
        return conversations

    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Return mock data for development if real data fails completely
        return [
            {
                "conversation_id": f"mock_{i}",
                "conversation": [
                    {"role": "user", "content": f"Mock Question {i}: What is the capital of France?"},
                    {"role": "assistant", "content": "Paris"},
                ],
                "original_idx": i,
            }
            for i in range(5)
        ]

def split_conversation_into_tasks(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split conversations into individual turn tasks.
    We focus on the USER's last question to generate a response.
    Or, simulate a full conversation turn-by-turn.
    For this 'baseline', let's extracting the first user query or each user query as a task.
    """
    tasks = []
    for conv in conversations:
        cid = conv['conversation_id']
        messages = conv['conversation']
        
        # Simple extraction: Find all user messages and prepare a task to respond to them
        # given the history up to that point.
        history = []
        turn_idx = 0
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            
            if role == 'user':
                task = {
                    "task_id": f"{cid}_turn_{turn_idx}",
                    "conversation_id": cid,
                    "history": list(history), # Copy history
                    "input": content,
                    "turn_index": turn_idx
                }
                tasks.append(task)
                turn_idx += 1
                
            # Update history for next turns
            history.append(msg)
            
    return tasks
