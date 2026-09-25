"""
Evaluation and Benchmark: TypeSafe Jev (System 1) vs. fastino/GLiNER2.5-Decide on Indian Financial News Dataset.
Evaluates multi-task zero-shot classification: Sentiment, Sector, and Intent/Urgency.
"""

import os
import time
import json
import requests
import torch
import numpy as np
from typing import Dict, Any, List
from datasets import load_dataset
from gliner2 import AutoExtractor

def load_env(env_path: str = ".env") -> Dict[str, str]:
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

def run_jev_vs_gliner_benchmark(num_samples: int = 250, eval_all_gliner: int = 1000):
    env = load_env(".env")
    jev_key = env.get("JEV_API_KEY")
    gliner_path = env.get("GLINER_MODEL_PATH", "fastino/GLiNER2.5-Decide")
    
    print(f"Loading GLiNER2.5-Decide from {gliner_path}...")
    gliner_model = AutoExtractor.from_pretrained(gliner_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(gliner_model, "to"):
        gliner_model.to(device)
    elif hasattr(gliner_model, "model") and hasattr(gliner_model.model, "to"):
        gliner_model.model.to(device)
    print(f"GLiNER2.5-Decide loaded on device: {device}")
    
    jev_url = "https://api.typesafe.ai/v1/systemone"
    jev_headers = {"Authorization": f"Bearer {jev_key}", "Content-Type": "application/json"}
    
    print("\nLoading dataset `kdave/Indian_Financial_News` from Hugging Face...")
    dataset = load_dataset("kdave/Indian_Financial_News", split="train")
    total_data = len(dataset)
    print(f"Total dataset size: {total_data} articles.")
    
    # Step sampling for uniform coverage
    step = max(1, total_data // eval_all_gliner)
    sample_indices = [i * step for i in range(eval_all_gliner) if (i * step) < total_data][:eval_all_gliner]
    
    print(f"\nRunning benchmark: {num_samples} dual-model samples (Jev + GLiNER) and {eval_all_gliner} extended GLiNER samples...")
    print("=" * 100)
    
    # Categories
    sentiment_options = ["Positive", "Negative", "Neutral"]
    sector_options = [
        "Banking & Financial Services",
        "Macroeconomics, RBI & Govt Policy",
        "Information Technology & Tech",
        "Automobile & Manufacturing",
        "Energy & Infrastructure",
        "Healthcare & Pharmaceuticals",
        "FMCG, Retail & Consumer"
    ]
    
    # Metrics
    gliner_metrics = {
        "correct_sentiment": 0,
        "total_sentiment": 0,
        "latencies_ms": [],
        "sectors": {},
    }
    jev_metrics = {
        "correct_sentiment": 0,
        "total_sentiment": 0,
        "latencies_ms": [],
        "sectors": {},
        "input_tokens": 0,
        "cost_usd": 0.0
    }
    
    t_start_total = time.time()
    
    for idx, sample_idx in enumerate(sample_indices, 1):
        item = dataset[sample_idx]
        title_or_summary = item.get("Summary", "").strip()
        full_content = item.get("Content", "").strip()
        gt_sentiment = item.get("Sentiment", "").strip().capitalize()
        
        snippet = f"{title_or_summary} {full_content[:250]}" if full_content else title_or_summary
        
        # 1. Evaluate GLiNER2.5-Decide
        t0 = time.perf_counter()
        gliner_res = gliner_model.classify_text(
            snippet,
            {
                "sentiment": sentiment_options,
                "sector": sector_options
            }
        )
        gliner_lat = (time.perf_counter() - t0) * 1000.0
        gliner_metrics["latencies_ms"].append(gliner_lat)
        
        gliner_sent = gliner_res.get("sentiment", "Neutral")
        gliner_sec = gliner_res.get("sector", "Macroeconomics, RBI & Govt Policy")
        gliner_metrics["sectors"][gliner_sec] = gliner_metrics["sectors"].get(gliner_sec, 0) + 1
        
        if gt_sentiment in sentiment_options:
            gliner_metrics["total_sentiment"] += 1
            if gliner_sent == gt_sentiment:
                gliner_metrics["correct_sentiment"] += 1
                
        # 2. Evaluate TypeSafe Jev (if within dual-eval budget)
        jev_sent = None
        jev_sec = None
        jev_lat = 0.0
        
        if idx <= num_samples and jev_key:
            jev_questions = {
                "sentiment": {
                    "type": "choice",
                    "instructions": "Determine the financial and market sentiment of this Indian financial news article.",
                    "criteria": {opt: None for opt in sentiment_options}
                },
                "sector": {
                    "type": "choice",
                    "instructions": "Classify which Indian economic sector this news article directly concerns.",
                    "criteria": {opt: None for opt in sector_options}
                },
                "regulatory_action": {
                    "type": "noul",
                    "instructions": "Does this development involve formal central bank (RBI), SEBI, or regulatory intervention?"
                }
            }
            
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    jev_url,
                    headers=jev_headers,
                    json={"state": snippet, "model": "jev-latest", "questions": jev_questions},
                    timeout=20
                )
                jev_lat = (time.perf_counter() - t0) * 1000.0
                jev_data = resp.json()
                
                if "answers" in jev_data:
                    jev_metrics["latencies_ms"].append(jev_lat)
                    in_tok = jev_data.get("usage", {}).get("input_tokens", 350)
                    jev_metrics["input_tokens"] += in_tok
                    jev_metrics["cost_usd"] += in_tok * (0.042 / 1_000_000)
                    
                    jev_sent = jev_data["answers"]["sentiment"]["choice"]
                    jev_sec = jev_data["answers"]["sector"]["choice"]
                    jev_metrics["sectors"][jev_sec] = jev_metrics["sectors"].get(jev_sec, 0) + 1
                    
                    if gt_sentiment in sentiment_options:
                        jev_metrics["total_sentiment"] += 1
                        if jev_sent == gt_sentiment:
                            jev_metrics["correct_sentiment"] += 1
            except Exception as e:
                print(f"Jev API error on sample {sample_idx}: {e}")
                
        # Print progress update
        if idx % 25 == 0 or idx == 1 or idx == num_samples:
            print(f"[{idx:04d}/{len(sample_indices):04d}] Sample #{sample_idx:<5} | GLiNER: {gliner_lat:.1f}ms (Sent: {gliner_sent}) | Jev: {jev_lat:.1f}ms (Sent: {jev_sent}) [GT: {gt_sentiment}]")
            
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS & METRICS SUMMARY")
    print("=" * 100)
    
    # GLiNER summary
    gl_acc = (gliner_metrics["correct_sentiment"] / max(1, gliner_metrics["total_sentiment"])) * 100.0
    gl_mean_lat = sum(gliner_metrics["latencies_ms"]) / len(gliner_metrics["latencies_ms"])
    gl_med_lat = sorted(gliner_metrics["latencies_ms"])[len(gliner_metrics["latencies_ms"]) // 2]
    gl_throughput = len(gliner_metrics["latencies_ms"]) / (sum(gliner_metrics["latencies_ms"]) / 1000.0)
    
    print(f"► fastino/GLiNER2.5-Decide (340M Local GPU, N={len(gliner_metrics['latencies_ms'])})")
    print(f"  • Sentiment Zero-Shot Accuracy: {gliner_metrics['correct_sentiment']}/{gliner_metrics['total_sentiment']} ({gl_acc:.2f}%)")
    print(f"  • Latency (Mean / Median):      {gl_mean_lat:.2f} ms / {gl_med_lat:.2f} ms")
    print(f"  • Throughput:                   {gl_throughput:.1f} articles/sec")
    print(f"  • Cost:                         $0.000000 (Local / Air-Gapped)")
    
    if jev_metrics["total_sentiment"] > 0:
        j_acc = (jev_metrics["correct_sentiment"] / jev_metrics["total_sentiment"]) * 100.0
        j_mean_lat = sum(jev_metrics["latencies_ms"]) / len(jev_metrics["latencies_ms"])
        j_med_lat = sorted(jev_metrics["latencies_ms"])[len(jev_metrics["latencies_ms"]) // 2]
        print(f"\n► TypeSafe Jev (Cloud System 1, N={len(jev_metrics['latencies_ms'])})")
        print(f"  • Sentiment Zero-Shot Accuracy: {jev_metrics['correct_sentiment']}/{jev_metrics['total_sentiment']} ({j_acc:.2f}%)")
        print(f"  • Latency (Mean / Median):      {j_mean_lat:.2f} ms / {j_med_lat:.2f} ms")
        print(f"  • Total Tokens / Spend:         {jev_metrics['input_tokens']} tokens / ${jev_metrics['cost_usd']:.6f} USD")
        
    print("\nGLiNER Sector Classification Distribution:")
    for sec, count in sorted(gliner_metrics["sectors"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {sec:<36}: {count:4d} ({count/len(gliner_metrics['latencies_ms']):.1%})")
        
    print("=" * 100)

if __name__ == "__main__":
    run_jev_vs_gliner_benchmark(num_samples=100, eval_all_gliner=500)
