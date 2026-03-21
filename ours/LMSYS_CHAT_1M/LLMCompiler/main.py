import argparse
import json
import os
import logging
import time
from tqdm import tqdm

from config import (
    LARGE_MODEL_API_KEY, LARGE_MODEL_API_BASE, LARGE_MODEL_NAME,
    SMALL_MODEL_API_KEY, SMALL_MODEL_API_BASE, SMALL_MODEL_NAME,
    SAMPLE_SIZE_Multi, SAMPLE_SIZE_Single, EXPERIENCE_SIMILARITY_THRESHOLD
)
from dataset_utils import load_lmsys_chat_1m, split_conversation_into_tasks
from llm_client import LLMClient
from agent import LLMCompileAgent
from evaluation import Evaluator
from experience_db import ExperienceDB, analyze_task, generate_experience_summary

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
    
    if mode != 'hybrid':
        raise ValueError(f"Only 'hybrid' mode is supported in the current optimized implementation.")
        
    evaluator = Evaluator() if eval_enabled else None
    db = ExperienceDB()
    
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
                
                used_experience = ""
                routing_large_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                
                # 1. Analyze task complexity
                # analyze_task always uses small model
                small_calls = 1
                large_calls = 0
                
                # Tracking extra transfer bytes in main.py logic
                main_s2l_bytes = 0
                main_l2s_bytes = 0
                
                refined_query, category, _ = analyze_task(small_llm, user_query, history)
                
                if category == 1:
                    # Simple task -> Small model
                    logger.info(f"Task {task['task_id']} is SIMPLE. Using small model.")
                    current_agent = LLMCompileAgent(plan_llm=small_llm, exec_llm=small_llm)
                    current_method = "small_only (simple)"
                else:
                    # Complex task -> Search DB
                    main_s2l_bytes += len(refined_query.encode('utf-8'))
                    found, exp_data = db.search(refined_query, EXPERIENCE_SIMILARITY_THRESHOLD)
                    if found:
                        logger.info(f"Task {task['task_id']} is COMPLEX. Experience found (score: {exp_data['score']:.2f}). Using small model with experience.")
                        used_experience = exp_data.get("experience", "")
                        main_l2s_bytes += len(used_experience.encode('utf-8'))
                        current_agent = LLMCompileAgent(plan_llm=small_llm, exec_llm=small_llm)
                        current_method = "small_only (experience_retrieved)"
                    else:
                        logger.info(f"Task {task['task_id']} is COMPLEX. No experience found. Using large model.")
                        current_agent = LLMCompileAgent(plan_llm=large_llm, exec_llm=large_llm)
                        current_method = "large_only (fallback)"
                
                result = current_agent.run(user_query, history, experience=used_experience)
                
                end_time = time.time()
                latency = end_time - start_time
                
                # Re-calculate calls and token usage based on the selected method from result.total_calls
                agent_total_calls = result.get("total_calls", 0)
                if "large_only" in current_method:
                    large_calls += agent_total_calls
                    usage = result.get("usage", {})
                else:
                    small_calls += agent_total_calls
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                
                # Token usage logic (from agent execution)
                usage_prompt = usage.get("prompt_tokens", 0)
                usage_completion = usage.get("completion_tokens", 0)
                usage_total = usage.get("total_tokens", 0)
                
                total_usage = {"prompt_tokens": usage_prompt, "completion_tokens": usage_completion, "total_tokens": usage_total}
                answer_content = result.get('answer', '')
                
                llm_score = 0.0
                if evaluator and answer_content:
                    eval_res = evaluator.evaluate(user_query, answer_content, history)
                    llm_score = eval_res.get('llm_score', 0.0)
                
                # 2. Learn from large model successful traces
                if current_method == "large_only (fallback)" and llm_score > 8.0:
                    logger.info(f"Task {task['task_id']} (Large Model) scored {llm_score}. Generating experience summary...")
                    # Filter out large tool outputs to reduce context overhead for summary generation
                    filtered_trace = [s for s in result.get('trace', []) if "Tool Output" not in s.get('step', '')]
                    exp_summary, summary_usage = generate_experience_summary(large_llm, user_query, filtered_trace, history)
                    for k in routing_large_usage: routing_large_usage[k] += summary_usage.get(k, 0)
                    large_calls += 1
                    
                    if exp_summary:
                        db.add_experience(user_query, result.get('trace', []), exp_summary)
                        logger.info(f"Experience added to DB for Task {task['task_id']}.")
                
                # Combine routing/summary usage with agent usage
                for k in total_usage:
                    total_usage[k] += routing_large_usage.get(k, 0)
                    
                if "large_only" in current_method:
                    transfer_s2l = result.get('transfer_stats', {}).get('s2l', 0) + main_s2l_bytes
                    transfer_l2s = result.get('transfer_stats', {}).get('l2s', 0) + main_l2s_bytes
                else:
                    transfer_s2l = main_s2l_bytes
                    transfer_l2s = main_l2s_bytes
                
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
                    "model": "hybrid",
                    "method": current_method,
                    "details": {
                        "small_calls": small_calls,
                        "large_calls": large_calls,
                        "edge_to_cloud_kb": round(transfer_s2l / 1024, 3),
                        "cloud_to_edge_kb": round(transfer_l2s / 1024, 3)
                    }
                }

                f_out.write(json.dumps(full_result) + "\n")
                f_out.flush()
                
                stats["processed_count"] += 1
                stats["total_latency"] += latency
                stats["total_large_calls"] += large_calls
                stats["total_small_calls"] += small_calls
                stats["total_prompt_tokens"] += total_usage["prompt_tokens"]
                stats["total_completion_tokens"] += total_usage["completion_tokens"]
                stats["total_tokens"] += total_usage["total_tokens"]
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
    parser.add_argument("--mode", type=str, choices=['hybrid'], default='hybrid',
                        help="Execution mode (currently only 'hybrid' is supported)")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of conversations to load (default uses Multi=300, Single=100)")
    parser.add_argument("--no_eval", action="store_true", help="Disable LLM judge evaluation")
    
    args = parser.parse_args()
    
    run_pipeline(args.mode, args.sample_size, not args.no_eval)
