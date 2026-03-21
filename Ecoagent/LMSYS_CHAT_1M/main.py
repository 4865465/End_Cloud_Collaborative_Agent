import argparse
import json
import os
import logging
import time
from tqdm import tqdm

from config import (
    LARGE_MODEL_API_KEY, LARGE_MODEL_API_BASE, LARGE_MODEL_NAME,
    SMALL_MODEL_API_KEY, SMALL_MODEL_API_BASE, SMALL_MODEL_NAME,
    SAMPLE_SIZE_Multi, SAMPLE_SIZE_Single
)
from dataset_utils import load_lmsys_chat_1m, split_conversation_into_tasks
from llm_client import LLMClient
from agent import LLMCompileAgent
from evaluation import Evaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_progress(output_file: str, stats: dict) -> set:
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['task_id'])
                    
                    stats["processed_count"] += 1
                    metrics = data.get("metrics", {})
                    stats["total_latency"] += metrics.get("latency_seconds", 0)
                    stats["total_large_calls"] += metrics.get("large_model_calls", 0)
                    stats["total_small_calls"] += metrics.get("small_model_calls", 0)
                    
                    tu = metrics.get("token_usage", {})
                    stats["total_prompt_tokens"] += tu.get("prompt_tokens", 0)
                    stats["total_completion_tokens"] += tu.get("completion_tokens", 0)
                    stats["total_tokens"] += tu.get("total_tokens", 0)
                    
                    stats["total_llm_score"] += metrics.get("llm_judge_score", 0)
                    stats["total_search_calls"] += metrics.get("search_calls", 0)
                    stats["total_code_tokens"] += metrics.get("code_tokens", 0)
                    
                    stats["total_edge_to_cloud_bytes"] += metrics.get("edge_to_cloud_kb", 0) * 1024
                    stats["total_cloud_to_edge_bytes"] += metrics.get("cloud_to_edge_kb", 0) * 1024

                except json.JSONDecodeError:
                    continue
    return processed_ids

def run_pipeline(mode: str, sample_size: int = None, eval_enabled: bool = True):
    logger.info(f"Starting LLMCompiler pipeline in mode: {mode}")
    
    # 1. Initialize Clients
    large_llm = LLMClient(LARGE_MODEL_API_KEY, LARGE_MODEL_API_BASE, LARGE_MODEL_NAME)
    small_llm = LLMClient(SMALL_MODEL_API_KEY, SMALL_MODEL_API_BASE, SMALL_MODEL_NAME)
    
    if mode == 'large_only':
        agent = LLMCompileAgent(plan_llm=large_llm, exec_llm=large_llm)
    elif mode == 'small_only':
        agent = LLMCompileAgent(plan_llm=small_llm, exec_llm=small_llm)
    elif mode == 'hybrid':
        # LLM1 (planner/joiner) is large, LLM2 (execution/inference tool) is small
        agent = LLMCompileAgent(plan_llm=large_llm, exec_llm=small_llm)
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    evaluator = Evaluator() if eval_enabled else None
    
    logger.info("Loading dataset...")
    conversations = load_lmsys_chat_1m(sample_size=sample_size)
    tasks = split_conversation_into_tasks(conversations)
    logger.info(f"Generated {len(tasks)} tasks.")
    
    stats = {
        "processed_count": 0,
        "total_latency": 0.0,
        "total_large_calls": 0,
        "total_small_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_llm_score": 0.0,
        "total_edge_to_cloud_bytes": 0,
        "total_cloud_to_edge_bytes": 0,
        "total_search_calls": 0,
        "total_code_tokens": 0
    }

    output_file = f"results_{mode}.jsonl"
    processed_ids = load_progress(output_file, stats)
    logger.info(f"Found {len(processed_ids)} processed tasks. Resuming...")
    
    tasks_to_run = [t for t in tasks if t['task_id'] not in processed_ids]
    logger.info(f"Remaining tasks: {len(tasks_to_run)}")

    with open(output_file, 'a') as f_out: # Append mode
        for task in tqdm(tasks_to_run, desc="Processing"):
            try:
                # Counters are reset inside agent.run()
                
                start_time = time.time()
                
                user_query = task['input']
                history = task['history']
                
                result = agent.run(user_query, history)
                
                end_time = time.time()
                latency = end_time - start_time
                
                # Fetch actual calls directly from the agent result
                small_calls = result.get("small_calls", 0)
                large_calls = result.get("large_calls", 0)
                
                # Token usage logic
                usage = result.get("usage", {})
                usage_prompt = usage.get("prompt_tokens", 0)
                usage_completion = usage.get("completion_tokens", 0)
                usage_total = usage.get("total_tokens", 0)
                
                if mode == 'small_only':
                    small_calls = large_calls + small_calls
                    large_calls = 0
                    usage_prompt = usage_completion = usage_total = 0
                elif mode == 'large_only':
                    large_calls = large_calls + small_calls
                    small_calls = 0
                
                total_usage = {"prompt_tokens": usage_prompt, "completion_tokens": usage_completion, "total_tokens": usage_total}
                answer_content = result.get('answer', '')
                
                llm_score = 0.0
                if evaluator and answer_content:
                    eval_res = evaluator.evaluate(user_query, answer_content, history)
                    llm_score = eval_res.get('llm_score', 0.0)
                    
                if mode == 'small_only':
                    transfer_s2l = 0
                    transfer_l2s = 0
                elif mode == 'large_only':
                    payload_messages = history + [{"role": "user", "content": user_query}]
                    try:
                        serialized = json.dumps(payload_messages, ensure_ascii=False)
                        transfer_s2l = len(serialized.encode('utf-8'))
                    except Exception:
                        transfer_s2l = len(str(payload_messages).encode('utf-8'))
                    transfer_l2s = len(str(answer_content).encode('utf-8')) if answer_content else 0
                elif mode == 'hybrid':
                    transfer_s2l = result.get('transfer_stats', {}).get('s2l', 0)
                    transfer_l2s = result.get('transfer_stats', {}).get('l2s', 0)
                
                full_result = {
                    "task_id": task['task_id'],
                    "question": user_query,
                    "answer": answer_content,
                    "metrics": {
                        "latency_seconds": round(latency, 2),
                        "large_model_calls": large_calls,
                        "small_model_calls": small_calls,
                        "token_usage": total_usage,
                        "llm_judge_score": llm_score,
                        "edge_to_cloud_kb": round(transfer_s2l / 1024, 3),
                        "cloud_to_edge_kb": round(transfer_l2s / 1024, 3),
                        "search_calls": result.get("search_calls", 0),
                        "code_tokens": result.get("code_tokens", 0)
                    },
                    "trace": result.get('trace', []),
                    "model": mode,
                    "method": "llmcompiler_hybrid" if mode == 'hybrid' else f"{mode}_only"
                }
                
                if mode == 'hybrid':
                    full_result["details"] = {
                        "small_calls": small_calls,
                        "large_calls": large_calls,
                        "edge_to_cloud_kb": round(transfer_s2l / 1024, 3),
                        "cloud_to_edge_kb": round(transfer_l2s / 1024, 3)
                    }

                f_out.write(json.dumps(full_result) + "\n")
                f_out.flush()
                
                stats["processed_count"] += 1
                stats["total_latency"] += latency
                stats["total_large_calls"] += large_calls
                stats["total_small_calls"] += small_calls
                stats["total_prompt_tokens"] += usage_prompt
                stats["total_completion_tokens"] += usage_completion
                stats["total_tokens"] += usage_total
                stats["total_llm_score"] += llm_score
                stats["total_edge_to_cloud_bytes"] += transfer_s2l
                stats["total_cloud_to_edge_bytes"] += transfer_l2s
                stats["total_search_calls"] += result.get("search_calls", 0)
                stats["total_code_tokens"] += result.get("code_tokens", 0)
                    
            except Exception as e:
                logger.error(f"Error processing task {task['task_id']}: {e}")
                
    count = stats["processed_count"]
    if count > 0:
        summary = {
            "mode": mode,
            "processed_count": count,
            "avg_latency": stats["total_latency"] / count,
            "avg_large_calls": stats["total_large_calls"] / count,
            "avg_small_calls": stats["total_small_calls"] / count,
            "avg_prompt_tokens": stats["total_prompt_tokens"] / count,
            "avg_completion_tokens": stats["total_completion_tokens"] / count,
            "avg_total_tokens": stats["total_tokens"] / count,
            "avg_llm_score": stats["total_llm_score"] / count,
            "avg_search_calls": stats["total_search_calls"] / count,
            "avg_code_tokens": stats["total_code_tokens"] / count,
            "avg_edge_to_cloud_kb": (stats["total_edge_to_cloud_bytes"] / count) / 1024,
            "avg_cloud_to_edge_kb": (stats["total_cloud_to_edge_bytes"] / count) / 1024
        }
        
        summary_file = f"summary_{mode}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Summary saved to {summary_file}")
        print(json.dumps(summary, indent=4))
    else:
        logger.info("No tasks processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLMCompiler Pipeline")
    parser.add_argument("--mode", type=str, choices=['large_only', 'small_only', 'hybrid'], required=True, 
                        help="Execution mode: large_only, small_only, hybrid")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of conversations to load (default uses Multi=300, Single=100)")
    parser.add_argument("--no_eval", action="store_true", help="Disable LLM judge evaluation")
    
    args = parser.parse_args()
    
    run_pipeline(args.mode, args.sample_size, not args.no_eval)
