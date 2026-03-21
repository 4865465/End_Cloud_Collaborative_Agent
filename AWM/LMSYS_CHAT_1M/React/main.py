import argparse
import json
import os
import logging
import time
from tqdm import tqdm
from config import (
    LARGE_MODEL_API_KEY, LARGE_MODEL_API_BASE, LARGE_MODEL_NAME,
    SMALL_MODEL_API_KEY, SMALL_MODEL_API_BASE, SMALL_MODEL_NAME,
    SIMILARITY_THRESHOLD_1, EXPERIENCE_SAVE_THRESHOLD
)
from dataset_utils import load_lmsys_chat_1m, split_conversation_into_tasks
from llm_client import LLMClient
from agent import ReactAgent
from evaluation import Evaluator
from experience_db import ExperienceDB, analyze_task, generate_experience_summary

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_progress(output_file: str, stats: dict) -> set:
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['task_id'])
                    
                    # Accumulate stats from existing records
                    m = data.get('metrics', {})
                    stats["processed_count"] += 1
                    stats["total_latency"] += m.get("latency_seconds", 0.0)
                    stats["total_large_calls"] += m.get("large_model_calls", 0)
                    stats["total_small_calls"] += m.get("small_model_calls", 0)
                    
                    usage = m.get("token_usage", {})
                    stats["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
                    stats["total_completion_tokens"] += usage.get("completion_tokens", 0)
                    stats["total_tokens"] += usage.get("total_tokens", 0)
                    
                    stats["total_llm_score"] += m.get("llm_judge_score", 0.0)
                    stats["total_search_calls"] += m.get("search_calls", 0)
                    stats["total_code_tokens"] += m.get("code_tokens", 0)
                    
                    stats["total_edge_to_cloud_bytes"] += m.get("edge_to_cloud_kb", 0.0) * 1024
                    stats["total_cloud_to_edge_bytes"] += m.get("cloud_to_edge_kb", 0.0) * 1024
                    
                except (json.JSONDecodeError, KeyError):
                    continue
    return processed_ids

def run_pipeline(sample_size: int = None, eval_enabled: bool = True):
    logger.info("Starting pipeline in hybrid mode (Experience DB based)")
    
    # 1. Initialize Clients
    large_llm = LLMClient(LARGE_MODEL_API_KEY, LARGE_MODEL_API_BASE, LARGE_MODEL_NAME)
    small_llm = LLMClient(SMALL_MODEL_API_KEY, SMALL_MODEL_API_BASE, SMALL_MODEL_NAME)
    
    # 2. Initialize Agents
    agent_large = ReactAgent(large_llm, is_edge=False)
    agent_small = ReactAgent(small_llm, is_edge=True)
    
    # 3. Initialize Evaluator & DB
    evaluator = Evaluator() if eval_enabled else None
    exp_db = ExperienceDB("experience_db.json")

    
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
        "total_edge_to_cloud_bytes": 0.0,
        "total_cloud_to_edge_bytes": 0.0,
    }

    # 5. Filter processed tasks
    output_file = "results.jsonl"
    processed_ids = load_progress(output_file, stats)
    logger.info(f"Found {len(processed_ids)} processed tasks. Resuming...")
    
    tasks_to_run = [t for t in tasks if t['task_id'] not in processed_ids]
    logger.info(f"Remaining tasks: {len(tasks_to_run)}")
    

    # 6. Process Queue
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for task in tqdm(tasks_to_run, desc="Processing"):
            try:
                start_time = time.time()
                
                user_query = task['input']
                history = task['history']
                
                # Metrics
                large_calls = 0
                small_calls = 0
                edge_to_cloud_bytes = 0
                cloud_to_edge_bytes = 0
                total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                refined_query, task_category, usage = analyze_task(small_llm, user_query, history)
                logger.info(f'Refined Query: {refined_query}, Category: {task_category}')

                # Metrics initialization
                result_info = {}
                final_res = {}
                exp_param = ""
                found = False

                if task_category == 1:
                    logger.info("Category 1: Simple task detected. Small model agent handles directly.")
                    final_res = agent_small.run(user_query, history)
                    small_calls += final_res.get('total_calls', 0)
                    result_info = {
                        "model": "small",
                        "method": "direct_small_agent",
                        "category": 1
                    }
                else:
                    # Complex task - Hybrid logic (Experience Retrieval + Fallback)
                    edge_to_cloud_bytes += len(refined_query.encode('utf-8'))
                    query_emb = exp_db.encoder.encode_single(refined_query)
                    found, best_match = exp_db.search_by_emb(query_emb, SIMILARITY_THRESHOLD_1)
                    
                    if found:
                        logger.info(f"Related experience found (Score {best_match['score']:.2f}). Passing to small model.")
                        cloud_to_edge_bytes += len(best_match['experience'].encode('utf-8'))
                        final_res = agent_small.run(user_query, history, experience=best_match['experience'].strip(), similar_query=best_match['query'])
                        small_calls += final_res.get('total_calls', 0)
                        result_info = {
                            "model": "small",
                            "method": "hybrid_small_with_exp",
                            "retrieved_score": float(best_match['score']),
                            "category": 0
                        }
                    else:
                        logger.info("No query above similarity threshold. Passing to large model.")
                        res_large = agent_large.run(user_query, history)
                        final_res = res_large
                        large_calls += res_large.get('total_calls', 0)
                        u = res_large.get('usage', {})
                        for k in total_usage: total_usage[k] += u.get(k, 0)
                        
                        result_info = {
                            "model": "large",
                            "method": "hybrid_large_fallback",
                            "category": 0
                        }

                end_time = time.time()
                latency = end_time - start_time

                # Integrate Agent Transfer Stats
                t_stats = final_res.get("transfer_stats", {"s2l": 0, "l2s": 0})
                edge_to_cloud_bytes += t_stats.get("s2l", 0)
                cloud_to_edge_bytes += t_stats.get("l2s", 0)

                # Extract Answer and Evaluation
                answer_content = final_res.get('answer', '')
                llm_score = 0.0
                if evaluator and answer_content:
                    eval_res = evaluator.evaluate(user_query, answer_content, history)
                    llm_score = eval_res.get('llm_score', 0.0)

                # Generate and save experience based on evaluation score (only for large model baseline path)
                if result_info.get("model") == "large" and answer_content and llm_score > EXPERIENCE_SAVE_THRESHOLD:
                    logger.info(f"Evaluation score {llm_score} > threshold {EXPERIENCE_SAVE_THRESHOLD}. Generating experience...")
                    trace = [t for t in final_res.get('trace', []) if t.get('type') not in ['observation', 'final']]
                    # print(trace)
                    if trace:
                        exp_desc, usage = generate_experience_summary(large_llm, user_query, trace, history)
                        large_calls += 1
                        for k in total_usage: total_usage[k] += usage.get(k, 0)
                        
                        if exp_desc:
                            trace1 = [t for t in final_res.get('trace', []) if t.get('type') not in ['observation']]
                            logger.info(f"Saving experience for query: {refined_query}")
                            exp_db.add_experience(refined_query, trace1, exp_desc)
                        else:
                            logger.info("Failed to generate experience summary.")
                elif result_info.get("model") == "large":
                    logger.info(f"Evaluation score {llm_score} < threshold {EXPERIENCE_SAVE_THRESHOLD}. Skipping experience saving.")



                # Assemble Final Result
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
                        "edge_to_cloud_kb": round(edge_to_cloud_bytes / 1024, 3),
                        "cloud_to_edge_kb": round(cloud_to_edge_bytes / 1024, 3),
                        "search_calls": final_res.get("search_calls", 0),
                        "code_tokens": final_res.get("code_tokens", 0)
                    },
                    "trace": final_res.get('trace', []),
                    **result_info
                }
                
                # Save result
                f_out.write(json.dumps(full_result, ensure_ascii=False) + "\n")
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
                stats["total_edge_to_cloud_bytes"] += edge_to_cloud_bytes
                stats["total_cloud_to_edge_bytes"] += cloud_to_edge_bytes
                stats["total_search_calls"] += final_res.get("search_calls", 0)
                stats["total_code_tokens"] += final_res.get("code_tokens", 0)
                    
            except Exception as e:
                logger.error(f"Error processing task {task['task_id']}: {e}", exc_info=True)
                
    # Calculate Summary
    count = stats["processed_count"]
    if count > 0:
        summary = {
            "mode": "hybrid",
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
        
        summary_file = "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Summary saved to {summary_file}")
        print(json.dumps(summary, indent=4))
    else:
        logger.info("No tasks processed successfully.")

    logger.info("Pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline1 Pipeline")
    parser.add_argument("--mode", type=str, choices=['hybrid'], default='hybrid',
                        help="Execution mode: hybrid (only hybrid supported now)")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of conversations to load (default uses Multi=300, Single=100)")
    parser.add_argument("--no_eval", action="store_true", help="Disable LLM judge evaluation")
    
    args = parser.parse_args()
    
    run_pipeline(args.sample_size, not args.no_eval)
