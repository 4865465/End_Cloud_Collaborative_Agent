import argparse
import json
import os
import logging
import time
from typing import List, Dict, Any
from tqdm import tqdm
from config import (
    LARGE_MODEL_API_KEY, LARGE_MODEL_API_BASE, LARGE_MODEL_NAME,
    SMALL_MODEL_API_KEY, SMALL_MODEL_API_BASE, SMALL_MODEL_NAME,
    SIMILARITY_THRESHOLD_1, SIMILARITY_THRESHOLD_2, SAMPLE_SIZE_Multi, SAMPLE_SIZE_Single, 
    FAILURE_SCORE_THRESHOLD, FAILURE_EXPERIENCE_UPDATE
)
from dataset_utils import load_lmsys_chat_1m, split_conversation_into_tasks
from llm_client import LLMClient
from agent import ReactAgent
from evaluation import Evaluator
from experience_db import ExperienceDB, reflect_on_failure, analyze_task
from tool_db import ToolDB

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
                    stats["total_code_tokens"] += m.get("code_tokens", m.get("code_input_tokens", 0) + m.get("code_output_tokens", 0))
                    
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
    
    tool_db_edge = ToolDB("edge_tool_db.json")
    tool_db_cloud = ToolDB("cloud_tool_db.json")
    # 2. Initialize Agents
    agent_large = ReactAgent(large_llm, is_edge=False, edge_tool_db=tool_db_edge, cloud_tool_db=tool_db_cloud, generate_experience=True)
    agent_small = ReactAgent(small_llm, is_edge=True, edge_tool_db=tool_db_edge, cloud_tool_db=tool_db_cloud)
    
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
    output_file = "results_ToolMemory.jsonl"
    processed_ids = load_progress(output_file, stats)
    logger.info(f"Found {len(processed_ids)} processed tasks. Resuming...")
    
    tasks_to_run = [t for t in tasks if t['task_id'] not in processed_ids]
    logger.info(f"Remaining tasks: {len(tasks_to_run)}")
    
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for task in tqdm(tasks_to_run, desc="Processing"):
            try:
                user_query = task['input']
                history = task['history']
                
                # Use unified single query runner
                full_result = run_single_query(
                    user_query=user_query,
                    history=history,
                    large_llm=large_llm,
                    small_llm=small_llm,
                    agent_large=agent_large,
                    agent_small=agent_small,
                    exp_db=exp_db,
                    evaluator=evaluator,
                    task_id=task['task_id']
                )

                # Save result
                f_out.write(json.dumps(full_result, ensure_ascii=False) + "\n")
                f_out.flush()
                
                # Update stats
                m = full_result["metrics"]
                stats["processed_count"] += 1
                stats["total_latency"] += m["latency_seconds"]
                stats["total_large_calls"] += m["large_model_calls"]
                stats["total_small_calls"] += m["small_model_calls"]
                stats["total_prompt_tokens"] += m["token_usage"].get("prompt_tokens", 0)
                stats["total_completion_tokens"] += m["token_usage"].get("completion_tokens", 0)
                stats["total_tokens"] += m["token_usage"].get("total_tokens", 0)
                stats["total_llm_score"] += m["llm_judge_score"]
                stats["total_search_calls"] += m["search_calls"]
                stats["total_code_tokens"] += m.get("code_tokens", m.get("code_input_tokens", 0) + m.get("code_output_tokens", 0))
                stats["total_edge_to_cloud_bytes"] += m["edge_to_cloud_kb"] * 1024
                stats["total_cloud_to_edge_bytes"] += m["cloud_to_edge_kb"] * 1024
                    
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
        
        summary_file = "summary_ToolMemory.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Summary saved to {summary_file}")
        print(json.dumps(summary, indent=4))
    else:
        logger.info("No tasks processed successfully.")
    logger.info("Pipeline completed.")

def run_single_query(
    user_query: str, 
    history: List[Dict[str, str]], 
    large_llm, 
    small_llm, 
    agent_large, 
    agent_small, 
    exp_db, 
    evaluator=None, 
    task_id=None,
    state_callback=None
) -> Dict[str, Any]:
    start_time = time.time()
    
    # 1. Metrics initialization
    large_calls = 0
    small_calls = 0
    edge_to_cloud_bytes = 0
    cloud_to_edge_bytes = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # 2. Preprocess/Analyze task complexity
    small_calls += 1
    refined_query, task_category, routing_usage = analyze_task(small_llm, user_query, history)
    
    result_info = {}
    final_res = {}
    exp_param = ""
    found = False
    best_match = {}

    if task_category == 1:
        # Case 1: Simple Agent task
        if state_callback: state_callback("Device", False) # Simple, not eligible for manual reflection
        res_small = agent_small.run(user_query, history)
        final_res = res_small
        small_calls += res_small.get('total_calls', 0)
        
        result_info = {
            "model": "small",
            "method": "direct_small_agent",
            "category": 1
        }
    else:
        # Case 2: Complex task - Hybrid logic
        edge_to_cloud_bytes += len(refined_query.encode('utf-8'))
        query_emb = exp_db.encoder.encode_single(refined_query)
        found, best_match = exp_db.search_by_emb(query_emb, SIMILARITY_THRESHOLD_1, SIMILARITY_THRESHOLD_2)
        
        if found:
            if state_callback: state_callback("Device", True) # Complex + Memory Hit, eligible for manual reflection
            reflection = best_match.get('reflection', '')
            reflection_str = f"[Previous Failure Reflection]:\n{reflection}\n\n" if reflection else ""
            
            if best_match.get("action") == "trace":
                trace_str = json.dumps(best_match['trace'], ensure_ascii=False)
                cloud_to_edge_bytes += len(trace_str.encode('utf-8'))
                exp_param = f"{trace_str}\n"
                method = "hybrid_small_with_trace"
            else:
                cloud_to_edge_bytes += len(best_match['experience'].encode('utf-8'))
                if reflection:
                    cloud_to_edge_bytes += len(reflection.encode('utf-8'))
                exp_param = f"{best_match['experience']}\n\n{reflection_str}"
                method = "hybrid_small_with_exp"
            
            result_info = {
                "model": "small",
                "method": method,
                "retrieved_score": float(best_match.get('score', 0)),
                "category": 0
            }
            res_small = agent_small.run(user_query, history, experience=exp_param.strip(), reference_type=("trace" if best_match.get("action") == "trace" else "experience"))
            final_res = res_small
            small_calls += res_small.get('total_calls', 0)
        else:
            # Fallback to large model
            if state_callback: state_callback("Cloud", False) # Cloud fallback, not eligible for manual reflection
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

    # Extract Answer and Experience
    answer_content = final_res.get('answer', '')
    exp_desc = final_res.get("experience_summary")
    
    # Run Evaluation (if provided)
    llm_score = 0.0
    if evaluator and answer_content:
        eval_res = evaluator.evaluate(user_query, answer_content, history)
        llm_score = eval_res.get('llm_score', 0.0)

    # Failure experience update (Innovation 3 reflection)
    if FAILURE_EXPERIENCE_UPDATE and evaluator and result_info.get("model") == "small" and result_info.get("category") == 0 and answer_content:
        if llm_score < FAILURE_SCORE_THRESHOLD:
            small_trace = [{k: v for k, v in t.items() if k != 'thought'} for t in final_res.get('trace', []) if t.get('type') != 'observation']
            ref_exp = exp_param.strip()
            reflection, usage = reflect_on_failure(large_llm, refined_query, ref_exp, small_trace, llm_score, history)
            large_calls += 1
            for k in total_usage: total_usage[k] += usage.get(k, 0)
            if reflection:
                edge_to_cloud_bytes += len(refined_query.encode('utf-8')) + len(json.dumps(small_trace, ensure_ascii=False).encode('utf-8'))
                exp_db.update_reflection(best_match.get('query'), reflection)

    # Experience update for large model fallback
    if result_info.get("model") == "large" and answer_content and exp_desc:
        trace = [{k: v for k, v in t.items() if k != 'thought'} for t in final_res.get('trace', []) if t.get('type') != 'observation']
        if trace:
            exp_db.add_experience(refined_query, trace, exp_desc)

    return {
        "task_id": task_id,
        "question": user_query,
        "refined_query": refined_query,
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
            "code_input_tokens": final_res.get("code_input_tokens", 0),
            "code_output_tokens": final_res.get("code_output_tokens", 0)
        },
        "trace": final_res.get('trace', []),
        "experience_summary": exp_desc,
        "used_experience": exp_param, # Useful for feedback reflection
        "exp_data": best_match,
        **result_info
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline1 Pipeline")
    parser.add_argument("--mode", type=str, choices=['hybrid'], default='hybrid',
                        help="Execution mode: hybrid (only hybrid supported now)")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of conversations to load (default uses Multi=300, Single=100)")
    parser.add_argument("--no_eval", action="store_true", help="Disable LLM judge evaluation")
    
    args = parser.parse_args()
    
    run_pipeline(args.sample_size, not args.no_eval)