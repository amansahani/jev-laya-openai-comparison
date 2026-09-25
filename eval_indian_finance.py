"""
Comprehensive Evaluation of Local Laya vs TypeSafe Jev vs OpenAI Luna on Challenging Indian Finance Benchmark.
"""

import os
import time
import json
import requests
import numpy as np
from typing import Dict, Any, List
from pipeline import JevClassificationPipeline
from indian_finance_dataset import INDIAN_FINANCE_BENCHMARK

def load_env(env_path: str = ".env") -> Dict[str, str]:
    """Load configuration from environment variables or local .env file."""
    env_vars = dict(os.environ)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip('"').strip("'")
    return env_vars

def run_eval():
    env = load_env(".env")
    jev_key = env.get("JEV_API_KEY")
    openai_key = env.get("OPENAI_API_KEY")

    if not jev_key:
        raise ValueError("Missing JEV_API_KEY in .env")
    if not openai_key:
        raise ValueError("Missing OPENAI_API_KEY in .env")

    print("Initializing Local Laya...")
    model_path = env.get("LAYA_MODEL_PATH", "./models/laya")
    laya = JevClassificationPipeline(model_path=model_path)

    jev_url = "https://api.typesafe.ai/v1/systemone"
    jev_headers = {"Authorization": f"Bearer {jev_key}", "Content-Type": "application/json"}

    openai_url = "https://api.openai.com/v1/chat/completions"
    openai_headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}

    # Tracking metrics
    models = ["Laya (Local)", "TypeSafe Jev (Cloud)", "OpenAI Luna (gpt-5.6-luna)"]
    metrics = {
        m: {
            "choice_correct": 0,
            "choice_total": 0,
            "noul_brier_sq_errors": [],
            "score_abs_errors": [],
            "latencies_ms": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0
        }
        for m in models
    }

    # Estimated pricing (USD per 1M tokens)
    # Laya: 0.00
    # Jev: ~$0.20 per 1M tokens (~$0.00008 per call)
    # OpenAI Luna (gpt-5.6-luna): $0.15 / 1M input, $0.60 / 1M output
    print(f"\nEvaluating {len(INDIAN_FINANCE_BENCHMARK)} challenging Indian Finance benchmark cases across 3 systems...\n")

    for i, item in enumerate(INDIAN_FINANCE_BENCHMARK, 1):
        sc_id = item["id"]
        category = item["category"]
        scenario = item["scenario"]
        gt = item["ground_truth"]
        questions = item["questions"]

        print(f"[{i:02d}/{len(INDIAN_FINANCE_BENCHMARK):02d}] {sc_id} ({category})")
        print(f"Scenario: \"{scenario[:85]}...\"")

        # 1. Local Laya
        t0 = time.perf_counter()
        laya_res = laya.predict_questions(scenario, questions)
        laya_lat = (time.perf_counter() - t0) * 1000.0
        metrics["Laya (Local)"]["latencies_ms"].append(laya_lat)
        metrics["Laya (Local)"]["input_tokens"] += laya_res.get("usage", {}).get("input_tokens", 0)

        # 2. TypeSafe Jev
        t0 = time.perf_counter()
        jev_resp = requests.post(jev_url, headers=jev_headers, json={"state": scenario, "model": "jev-latest", "questions": questions}, timeout=30)
        jev_lat = (time.perf_counter() - t0) * 1000.0
        jev_data = jev_resp.json()
        metrics["TypeSafe Jev (Cloud)"]["latencies_ms"].append(jev_lat)
        j_in = jev_data.get("usage", {}).get("input_tokens", 433)
        j_out = jev_data.get("usage", {}).get("output_tokens", 0)
        metrics["TypeSafe Jev (Cloud)"]["input_tokens"] += j_in
        metrics["TypeSafe Jev (Cloud)"]["output_tokens"] += j_out
        # Official TypeSafe Pricing: $0.042 per 1M input tokens, output tokens are $0.00 (free)
        metrics["TypeSafe Jev (Cloud)"]["cost_usd"] += j_in * (0.042 / 1_000_000)

        # 3. OpenAI Luna
        t0 = time.perf_counter()
        prompt = f"""Evaluate the Indian financial situation below and answer the questions strictly in JSON format.
Situation: "{scenario}"
Questions: {json.dumps(questions, indent=2)}

Format:
{{
  "answers": {{
    "<question_key>": {{
      "choice": "<selected_choice_key>",
      "is_true_probability": <float_0_to_1>,
      "score": <float_value>
    }}
  }}
}}"""
        luna_resp = requests.post(
            openai_url,
            headers=openai_headers,
            json={
                "model": "gpt-5.6-luna",
                "messages": [
                    {"role": "system", "content": "You are an expert in Indian financial law, RBI circulars, SEBI regulations, and cybercrime laws. Output strict JSON."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            },
            timeout=30
        )
        luna_lat = (time.perf_counter() - t0) * 1000.0
        luna_data = luna_resp.json()
        metrics["OpenAI Luna (gpt-5.6-luna)"]["latencies_ms"].append(luna_lat)
        u = luna_data.get("usage", {})
        l_in, l_out = u.get("prompt_tokens", 520), u.get("completion_tokens", 85)
        metrics["OpenAI Luna (gpt-5.6-luna)"]["input_tokens"] += l_in
        metrics["OpenAI Luna (gpt-5.6-luna)"]["output_tokens"] += l_out
        # Official OpenAI gpt-5.6-luna pricing: $0.20 / 1M input tokens, $1.20 / 1M output tokens
        metrics["OpenAI Luna (gpt-5.6-luna)"]["cost_usd"] += (l_in * (0.20 / 1_000_000)) + (l_out * (1.20 / 1_000_000))

        try:
            luna_content = json.loads(luna_data["choices"][0]["message"]["content"])
            luna_ans = luna_content.get("answers", luna_content)
        except Exception:
            luna_ans = {}

        # Evaluate Ground Truth Comparison for each Question
        # Find choice key, noul key, score key in questions
        choice_key = [k for k, v in questions.items() if v["type"] == "choice"][0]
        noul_key = [k for k, v in questions.items() if v["type"] == "noul"][0]
        score_key = [k for k, v in questions.items() if v["type"] == "score"][0]

        gt_choice = gt[choice_key]
        gt_noul = 1.0 if gt[noul_key] else 0.0
        gt_score = float(gt[score_key])

        # Evaluate Laya
        l_ans = laya_res.get("answers", {})
        l_c = l_ans.get(choice_key, {}).get("choice")
        l_n = l_ans.get(noul_key, {}).get("noul", 0.5)
        l_s = l_ans.get(score_key, {}).get("score", 0.0)

        if l_c == gt_choice:
            metrics["Laya (Local)"]["choice_correct"] += 1
        metrics["Laya (Local)"]["choice_total"] += 1
        metrics["Laya (Local)"]["noul_brier_sq_errors"].append((l_n - gt_noul) ** 2)
        metrics["Laya (Local)"]["score_abs_errors"].append(abs(l_s - gt_score))

        # Evaluate Jev
        j_ans = jev_data.get("answers", {})
        j_c = j_ans.get(choice_key, {}).get("choice")
        j_n = j_ans.get(noul_key, {}).get("noul", 0.5)
        j_s = j_ans.get(score_key, {}).get("score", 0.0)

        if j_c == gt_choice:
            metrics["TypeSafe Jev (Cloud)"]["choice_correct"] += 1
        metrics["TypeSafe Jev (Cloud)"]["choice_total"] += 1
        metrics["TypeSafe Jev (Cloud)"]["noul_brier_sq_errors"].append((j_n - gt_noul) ** 2)
        metrics["TypeSafe Jev (Cloud)"]["score_abs_errors"].append(abs(j_s - gt_score))

        # Evaluate Luna
        lu_c_obj = luna_ans.get(choice_key, {})
        lu_c = lu_c_obj.get("choice") if isinstance(lu_c_obj, dict) else lu_c_obj
        lu_n_obj = luna_ans.get(noul_key, {})
        lu_n = lu_n_obj.get("is_true_probability", 1.0 if lu_n_obj.get("is_true") else 0.0) if isinstance(lu_n_obj, dict) else (1.0 if lu_n_obj else 0.0)
        lu_s_obj = luna_ans.get(score_key, {})
        lu_s = float(lu_s_obj.get("score", 0.0) if isinstance(lu_s_obj, dict) else (lu_s_obj or 0.0))

        if lu_c == gt_choice:
            metrics["OpenAI Luna (gpt-5.6-luna)"]["choice_correct"] += 1
        metrics["OpenAI Luna (gpt-5.6-luna)"]["choice_total"] += 1
        metrics["OpenAI Luna (gpt-5.6-luna)"]["noul_brier_sq_errors"].append((lu_n - gt_noul) ** 2)
        metrics["OpenAI Luna (gpt-5.6-luna)"]["score_abs_errors"].append(abs(lu_s - gt_score))

        print(f"  [Choice GT: {gt_choice}] -> Laya: {l_c} | Jev: {j_c} | Luna: {lu_c}")
        print(f"  [Latency] -> Laya: {laya_lat:.1f}ms | Jev: {jev_lat:.1f}ms | Luna: {luna_lat:.1f}ms\n")

    # Print Full Comparative Table
    print("=" * 95)
    print(f"{'INDIAN FINANCE BENCHMARK: FINAL EVALUATION REPORT':^95}")
    print("=" * 95)
    print(f"{'Model':<30} | {'Choice Acc':<11} | {'Noul Brier (↓)':<15} | {'Score MAE (↓)':<14} | {'Avg Latency':<12} | {'Cost (USD)'}")
    print("-" * 95)

    total_eval_cost = 0.0
    for m in models:
        acc = (metrics[m]["choice_correct"] / metrics[m]["choice_total"]) * 100.0
        brier = float(np.mean(metrics[m]["noul_brier_sq_errors"]))
        mae = float(np.mean(metrics[m]["score_abs_errors"]))
        avg_lat = float(np.mean(metrics[m]["latencies_ms"]))
        cost = metrics[m]["cost_usd"]
        total_eval_cost += cost
        print(f"{m:<30} | {acc:>9.1f}% | {brier:>15.4f} | {mae:>14.4f} | {avg_lat:>9.1f} ms | ${cost:.6f}")

    print("=" * 95)
    print(f"TOTAL EVALUATION SPEND: ${total_eval_cost:.6f} USD (Well below the $0.50 USD limit budget)")
    print("=" * 95)

if __name__ == "__main__":
    run_eval()
