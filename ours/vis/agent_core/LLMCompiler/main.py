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
    SAMPLE_SIZE_Multi, SAMPLE_SIZE_Single, 
    SIMILARITY_THRESHOLD_1, SIMILARITY_THRESHOLD_2, 
    FAILURE_SCORE_THRESHOLD, FAILURE_EXPERIENCE_UPDATE
)
from dataset_utils import load_lmsys_chat_1m, split_conversation_into_tasks
from llm_client import LLMClient
from agent import LLMCompileAgent
from evaluation import Evaluator
from experience_db import ExperienceDB, analyze_task, reflect_on_failure

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
                    stats["total_code_tokens"] += metrics.get("code_tokens", metrics.get("code_input_tokens", 0) + metrics.get("code_output_tokens", 0))
                    
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
                user_query = task['input']
                history = task['history']
                
                full_result = run_single_query(
                    user_query=user_query,
                    history=history,
                    large_llm=large_llm,
                    small_llm=small_llm,
                    db=db,
                    evaluator=evaluator,
                    task_id=task['task_id']
                )

                f_out.write(json.dumps(full_result) + "\n")
                f_out.flush()
                
                m = full_result["metrics"]
                stats["processed_count"] += 1
                stats["total_latency"] += m["latency_seconds"]
                stats["total_large_calls"] += m["large_model_calls"]
                stats["total_small_calls"] += m["small_model_calls"]
                stats["total_prompt_tokens"] += m["token_usage"]['prompt_tokens']
                stats["total_completion_tokens"] += m["token_usage"]['completion_tokens']
                stats["total_tokens"] += m["token_usage"]['total_tokens']
                stats["total_llm_score"] += m["llm_judge_score"]
                stats["total_edge_to_cloud_bytes"] += m["edge_to_cloud_kb"] * 1024
                stats["total_cloud_to_edge_bytes"] += m["cloud_to_edge_kb"] * 1024
                stats["total_search_calls"] += m["search_calls"]
                stats["total_code_tokens"] += m.get("code_tokens", m.get("code_input_tokens", 0) + m.get("code_output_tokens", 0))
                    
            except Exception as e:
                logger.error(f"Error processing task {task['task_id']}: {e}")

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
    logger.info("Pipeline completed.")

def run_single_query(
    user_query: str,
    history: List[Dict[str, str]],
    large_llm,
    small_llm,
    db,
    evaluator=None,
    task_id=None,
    state_callback=None
) -> Dict[str, Any]:
    start_time = time.time()
    
    used_experience = ""
    exp_data = None
    routing_large_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    small_calls = 1
    large_calls = 0
    main_s2l_bytes = 0
    main_l2s_bytes = 0
    
    refined_query, category, routing_usage = analyze_task(small_llm, user_query, history)
    
    if category == 1:
        logger.info(f"Task is SIMPLE. Using small model.")
        if state_callback: state_callback("Device", False) # Simple, not eligible for manual reflection
        current_agent = LLMCompileAgent(plan_llm=small_llm, exec_llm=small_llm, generate_experience=False)
        current_method = "small_only (simple)"
    else:
        main_s2l_bytes += len(refined_query.encode('utf-8'))
        found, exp_data = db.search(refined_query, SIMILARITY_THRESHOLD_1, SIMILARITY_THRESHOLD_2)
        if found:
            if state_callback: state_callback("Device", True) # Complex + Memory Hit, eligible for manual reflection
            current_agent = LLMCompileAgent(plan_llm=small_llm, exec_llm=small_llm, generate_experience=False)
            if exp_data["action"] == "trace":
                raw_trace = exp_data.get("trace", [])
                filtered_trace = [s for s in raw_trace if s.get("step") in ["Planner Output", "Final Answer"]]
                trace_json = json.dumps(filtered_trace, ensure_ascii=False)
                used_experience = f"Action: trace\nTrace: {trace_json}"
                main_l2s_bytes += len(used_experience.encode('utf-8'))
                current_method = "small_only (hierarchical_trace_retrieved)"
            else:
                summary = exp_data.get("experience", "")
                reflection = exp_data.get("reflection", "")
                used_experience = f"Action: experience\nSummary: {summary}"
                if reflection:
                    used_experience += f"\nReflection: {reflection}"
                main_l2s_bytes += len(used_experience.encode('utf-8'))
                current_method = "small_only (hierarchical_exp_retrieved)"
        else:
            logger.info(f"Task is COMPLEX. No experience found. Using large model.")
            if state_callback: state_callback("Cloud", False) # Cloud fallback, not eligible for manual reflection
            current_agent = LLMCompileAgent(plan_llm=large_llm, exec_llm=large_llm, generate_experience=True)
            current_method = "large_only (fallback)"

    result = current_agent.run(user_query, history, experience=used_experience)
    
    end_time = time.time()
    latency = end_time - start_time
    
    agent_total_calls = result.get("total_calls", 0)
    if "large_only" in current_method:
        large_calls += agent_total_calls
        usage = result.get("usage", {})
    else:
        small_calls += agent_total_calls
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    total_usage = {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)}
    answer_content = result.get('answer', '')
    
    llm_score = 0.0
    if evaluator and answer_content:
        eval_res = evaluator.evaluate(user_query, answer_content, history)
        llm_score = eval_res.get('llm_score', 0.0)
    
    # Learn from large model
    if current_method == "large_only (fallback)":
        exp_summary = result.get('experience_summary')
        if exp_summary:
            db.add_experience(user_query, result.get('trace', []), exp_summary)

    # Failure experience update
    if FAILURE_EXPERIENCE_UPDATE and "hierarchical" in current_method and evaluator and llm_score < FAILURE_SCORE_THRESHOLD:
        ref_exp_content = used_experience
        reflection_text, ref_usage = reflect_on_failure(large_llm, user_query, ref_exp_content, result.get('trace', []), llm_score, history)
        if reflection_text:
            db.update_reflection(exp_data["query"], reflection_text)
            for k in routing_large_usage:
                routing_large_usage[k] += ref_usage.get(k, 0)
            large_calls += 1
    
    for k in total_usage:
        total_usage[k] += routing_large_usage.get(k, 0)
        
    transfer_stats = result.get('transfer_stats', {"s2l": 0, "l2s": 0})
    if "large_only" in current_method:
        transfer_s2l = transfer_stats.get('s2l', 0) + main_s2l_bytes
        transfer_l2s = transfer_stats.get('l2s', 0) + main_l2s_bytes
    else:
        transfer_s2l = main_s2l_bytes
        transfer_l2s = main_l2s_bytes
    
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
            "edge_to_cloud_kb": round(transfer_s2l / 1024, 3),
            "cloud_to_edge_kb": round(transfer_l2s / 1024, 3),
            "search_calls": result.get("search_calls", 0),
            "code_input_tokens": result.get("code_input_tokens", 0),
            "code_output_tokens": result.get("code_output_tokens", 0)
        },
        "trace": result.get('trace', []),
        "model": "hybrid",
        "method": current_method,
        "category": category,
        "used_experience": used_experience,
        "exp_data": exp_data
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLMCompiler Pipeline")
    parser.add_argument("--mode", type=str, choices=['hybrid'], default='hybrid',
                        help="Execution mode (currently only 'hybrid' is supported)")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of conversations to load (default uses Multi=300, Single=100)")
    parser.add_argument("--no_eval", action="store_true", help="Disable LLM judge evaluation")
    
    args = parser.parse_args()
    
    run_pipeline(args.mode, args.sample_size, not args.no_eval)
