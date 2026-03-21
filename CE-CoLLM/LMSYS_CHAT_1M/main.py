import argparse
import json
import os
import logging
import time
from tqdm import tqdm
from config import (
    LARGE_MODEL_API_KEY, LARGE_MODEL_API_BASE, LARGE_MODEL_NAME,
    SMALL_MODEL_API_KEY, SMALL_MODEL_API_BASE, SMALL_MODEL_NAME,
    CONFIDENCE_THRESHOLD, SAMPLE_SIZE_Multi, SAMPLE_SIZE_Single
)
from dataset_utils import load_lmsys_chat_1m, split_conversation_into_tasks
from llm_client import LLMClient
from agent import ReactAgent
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
    logger.info(f"Starting pipeline in mode: {mode}")
    
    # 1. Initialize Clients
    large_llm = LLMClient(LARGE_MODEL_API_KEY, LARGE_MODEL_API_BASE, LARGE_MODEL_NAME)
    small_llm = LLMClient(SMALL_MODEL_API_KEY, SMALL_MODEL_API_BASE, SMALL_MODEL_NAME)
    
    # 2. Initialize Agents
    agent_large = ReactAgent(large_llm)
    agent_small = ReactAgent(small_llm)
    
    # 3. Initialize Evaluator
    evaluator = Evaluator() if eval_enabled else None
    
    # 4. Load Dataset
    logger.info("Loading dataset...")
    conversations = load_lmsys_chat_1m(sample_size=sample_size)
    tasks = split_conversation_into_tasks(conversations)
    logger.info(f"Generated {len(tasks)} tasks.")
    
    # Stats accumulation
    stats = {
        "processed_count": 0,
        "total_latency": 0.0,
        "total_large_calls": 0,
        "total_small_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_llm_score": 0.0,
        "total_search_calls": 0,
        "total_code_tokens": 0,
        # 端↔云通信量（精确字节数）
        "total_edge_to_cloud_bytes": 0,
        "total_cloud_to_edge_bytes": 0,
    }
    
    # 5. Filter processed tasks
    output_file = f"results_{mode}.jsonl"
    processed_ids = load_progress(output_file, stats)
    logger.info(f"Found {len(processed_ids)} processed tasks. Resuming...")
    
    tasks_to_run = [t for t in tasks if t['task_id'] not in processed_ids]
    logger.info(f"Remaining tasks: {len(tasks_to_run)}")

    # 6. Process Queue
    with open(output_file, 'a') as f_out: # Append mode
        for task in tqdm(tasks_to_run, desc="Processing"):
            try:
                start_time = time.time()
                
                result = None
                user_query = task['input']
                history = task['history']
                
                # Metrics
                large_calls = 0
                small_calls = 0
                total_search_calls = 0
                total_code_tokens = 0
                total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                final_res = {} # To hold the final agent response wrapper

                if mode == 'large_only':
                    res = agent_large.run(user_query, history)
                    final_res = res
                    large_calls += res.get('total_steps', 0)
                    total_search_calls += res.get('search_calls', 0)
                    total_code_tokens += res.get('code_tokens', 0)
                    
                    # Add usage
                    u = res.get('usage', {})
                    for k in total_usage: total_usage[k] += u.get(k, 0)
                    
                    result = {
                        "model": "large",
                        "method": "large_only"
                    }
                    
                elif mode == 'small_only':
                    res = agent_small.run(user_query, history)
                    final_res = res
                    small_calls += res.get('total_steps', 0)
                    total_search_calls += res.get('search_calls', 0)
                    total_code_tokens += res.get('code_tokens', 0)
                    
                    # 我们只关心云端开销，这里不将小模型的Token计入统计

                    result = {
                        "model": "small",
                        "method": "small_only"
                    }
                    
                elif mode == 'hybrid':
                    res_small = agent_small.run(user_query, history)
                    small_calls += res_small.get('total_steps', 0)
                    total_search_calls += res_small.get('search_calls', 0)
                    total_code_tokens += res_small.get('code_tokens', 0)
                    
                    # 我们只关心云端开销，这里不将小模型的Token计入统计
                    
                    conf = res_small.get('confidence', 0.0)
                    print('small model confidence:', conf)
                    if conf > CONFIDENCE_THRESHOLD:
                        final_res = res_small
                        result = {
                            "model": "small",
                            "method": "hybrid_small_pass"
                        }
                    else:
                        logger.info(f"Small model confidence {conf:.2f} < {CONFIDENCE_THRESHOLD}. Fallback.")
                        res_large = agent_large.run(user_query, history)
                        final_res = res_large
                        large_calls += res_large.get('total_steps', 0)
                        total_search_calls += res_large.get('search_calls', 0)
                        total_code_tokens += res_large.get('code_tokens', 0)
                        
                        u = res_large.get('usage', {})
                        for k in total_usage: total_usage[k] += u.get(k, 0)
                        
                        result = {
                            "model": "large",
                            "method": "hybrid_large_fallback",
                            "small_model_trace": res_small.get('trace')
                        }

                end_time = time.time()
                latency = end_time - start_time
                
                # Extract Answer
                answer_content = final_res.get('answer', '')
                
                # Run Evaluation
                llm_score = 0.0
                if evaluator and answer_content:
                    # Evaluate answer against question (reference-free / context-aware)
                    eval_res = evaluator.evaluate(user_query, answer_content, history)
                    print(f'eval_res:{eval_res}')
                    llm_score = eval_res.get('llm_score', 0.0)

                # Assemble Final Result
                transfer_stats = final_res.get("transfer_stats", {})
                
                # 基于假设：若为小模型本地处理（small_only / hybrid_small_pass），则并不消耗端云通信开销
                if result.get("model") == "small":
                    actual_s2l = 0
                    actual_l2s = 0
                else:
                    actual_s2l = transfer_stats.get("s2l", 0)
                    actual_l2s = transfer_stats.get("l2s", 0)

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
                        "search_calls": total_search_calls,
                        "code_tokens": total_code_tokens,
                        "edge_to_cloud_kb": round(actual_s2l / 1024, 3),
                        "cloud_to_edge_kb": round(actual_l2s / 1024, 3),
                    },
                    "trace": final_res.get('trace', []),
                    **result # Merges model/method info
                }
                
                # Save result
                f_out.write(json.dumps(full_result) + "\n")
                f_out.flush()
                
                # Update stats
                stats["processed_count"] += 1
                stats["total_latency"] += latency
                stats["total_large_calls"] += large_calls
                stats["total_small_calls"] += small_calls
                stats["total_prompt_tokens"] += total_usage.get("prompt_tokens", 0)
                stats["total_completion_tokens"] += total_usage.get("completion_tokens", 0)
                stats["total_tokens"] += total_usage.get("total_tokens", 0)
                stats["total_llm_score"] += llm_score
                stats["total_search_calls"] += total_search_calls
                stats["total_code_tokens"] += total_code_tokens
                # 累积端↔云通信量（精确字节数）
                stats["total_edge_to_cloud_bytes"] += actual_s2l
                stats["total_cloud_to_edge_bytes"] += actual_l2s
                    
            except Exception as e:
                logger.error(f"Error processing task {task['task_id']}: {e}")
                # Don't increment stats for failed tasks
                
    # Calculate Summary
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
            # 平均端↔云通信量（通过字节计算后转为KB输出）
            "avg_edge_to_cloud_kb": (stats["total_edge_to_cloud_bytes"] / count) / 1024,
            "avg_cloud_to_edge_kb": (stats["total_cloud_to_edge_bytes"] / count) / 1024,
        }
        
        summary_file = f"summary_{mode}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Summary saved to {summary_file}")
        print(json.dumps(summary, indent=4))
    else:
        logger.info("No tasks processed successfully.")

    logger.info("Pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline1 Pipeline")
    parser.add_argument("--mode", type=str, choices=['large_only', 'small_only', 'hybrid'], required=True, 
                        help="Execution mode: large_only, small_only, hybrid")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of conversations to load (default uses Multi=300, Single=100)")
    parser.add_argument("--no_eval", action="store_true", help="Disable LLM judge evaluation")
    
    args = parser.parse_args()
    
    run_pipeline(args.mode, args.sample_size, not args.no_eval)
