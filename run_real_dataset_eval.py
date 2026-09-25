"""
Real-world Indian Financial News evaluation using `kdave/Indian_Financial_News` dataset.
Runs dynamic zero-shot classification on local Laya (and optionally TypeSafe Jev / OpenAI).
"""

import os
import time
import json
import torch
from typing import Dict, Any, List
from datasets import load_dataset
from pipeline import JevClassificationPipeline

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

def run_real_finance_eval(num_samples: int = 100):
    env = load_env(".env")
    model_path = env.get("LAYA_MODEL_PATH", "./models/laya")
    
    print(f"Loading local Laya pipeline from {model_path}...")
    pipeline = JevClassificationPipeline(model_path=model_path)
    
    print(f"\nLoading real Indian Financial News dataset (`kdave/Indian_Financial_News`)...")
    dataset = load_dataset("kdave/Indian_Financial_News", split="train")
    
    total_available = len(dataset)
    # Select 100 evenly distributed samples across the dataset
    step = max(1, total_available // num_samples)
    selected_indices = [i * step for i in range(num_samples) if (i * step) < total_available][:num_samples]
    
    print(f"Evaluating {len(selected_indices)} real-world Indian financial articles from dataset (N={total_available})...\n")
    print("=" * 100)
    
    results = []
    correct_sentiment = 0
    total_sentiment = 0
    latencies = []
    sector_counts = {}
    severity_scores = []
    regulatory_probs = []
    total_sentiment = 0
    latencies = []
    
    for idx, sample_idx in enumerate(selected_indices, 1):
        item = dataset[sample_idx]
        title_or_summary = item.get("Summary", "").strip()
        full_content = item.get("Content", "").strip()
        gt_sentiment = item.get("Sentiment", "").strip().capitalize()
        
        # Take the most informative snippet (summary + first 200 chars of content)
        snippet = f"{title_or_summary}\n\nContext: {full_content[:350]}..." if full_content else title_or_summary
        
        # Define dynamic zero-shot questions
        questions = {
            "market_sentiment": {
                "type": "choice",
                "question": "What is the primary financial and market sentiment of this Indian financial development?",
                "options": ["Positive", "Negative", "Neutral"]
            },
            "sector_classification": {
                "type": "choice",
                "question": "Which Indian economic sector is most directly impacted or discussed?",
                "options": [
                    "Banking & Financial Services",
                    "Information Technology & Tech",
                    "Automobile & Mobility",
                    "Energy & Infrastructure",
                    "FMCG, Retail & Consumer",
                    "Healthcare & Pharmaceuticals",
                    "Macroeconomics, RBI Policy & Govt"
                ]
            },
            "market_impact_severity": {
                "type": "score",
                "question": "Rate the financial market severity and volatility impact of this news on Indian markets (0=Negligible, 1=Minor/Sectoral, 2=Moderate, 3=High/Systemic):",
                "min": 0,
                "max": 3
            },
            "systemic_or_regulatory_action": {
                "type": "noul",
                "question": "Does this development involve formal regulatory, central bank (RBI), SEBI, or government policy intervention?",
                "options": ["True", "False"]
            }
        }
        
        t0 = time.perf_counter()
        pred = pipeline.predict_questions(snippet, questions)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        
        pred_sent = pred["answers"]["market_sentiment"]["prediction"]
        pred_sent_conf = pred["answers"]["market_sentiment"]["confidence"]
        pred_sector = pred["answers"]["sector_classification"]["prediction"]
        pred_severity = pred["answers"]["market_impact_severity"]["score"]
        pred_regulatory_prob = pred["answers"]["systemic_or_regulatory_action"]["probability"]
        
        is_match = (pred_sent == gt_sentiment) or (gt_sentiment not in ["Positive", "Negative", "Neutral"])
        if gt_sentiment in ["Positive", "Negative", "Neutral"]:
            total_sentiment += 1
            if pred_sent == gt_sentiment:
                correct_sentiment += 1
                
        results.append({
            "idx": sample_idx,
            "summary": title_or_summary[:80] + "...",
            "ground_truth_sentiment": gt_sentiment,
            "pred_sentiment": pred_sent,
            "sentiment_conf": pred_sent_conf,
            "pred_sector": pred_sector,
            "severity_score": pred_severity,
            "regulatory_prob": pred_regulatory_prob,
            "latency_ms": lat
        })
        
        sector_counts[pred_sector] = sector_counts.get(pred_sector, 0) + 1
        severity_scores.append(pred_severity)
        regulatory_probs.append(pred_regulatory_prob)
        
        # Progress indicator every 10 samples or on sample 1
        if idx % 10 == 0 or idx == 1:
            print(f"[{idx:03d}/{len(selected_indices):03d}] Sample #{sample_idx:<5} | Latency: {lat:.1f}ms")
            print(f"  Summary: {title_or_summary[:85]}...")
            print(f"  ► Sent: {pred_sent:<8} (Conf: {pred_sent_conf:.1%}) [GT: {gt_sentiment}] {'✓' if pred_sent == gt_sentiment else '✗'} | Sector: {pred_sector}")
            print(f"  ► Severity: {pred_severity:.2f}/3.00 | Reg. Prob: {pred_regulatory_prob:.1%}")
            print("-" * 100)
        
    print("\n" + "=" * 100)
    print(f"EVALUATION SUMMARY ON REAL INDIAN FINANCIAL NEWS (N={len(selected_indices)} Samples)")
    print("=" * 100)
    if total_sentiment > 0:
        acc = (correct_sentiment / total_sentiment) * 100.0
        print(f"Zero-Shot Sentiment Accuracy (vs Ground Truth): {correct_sentiment}/{total_sentiment} ({acc:.1f}%)")
    print(f"Average Multi-Task Latency (4 Questions/Item):  {sum(latencies)/len(latencies):.2f} ms")
    print(f"Median Latency:                                 {sorted(latencies)[len(latencies)//2]:.2f} ms")
    print(f"Min / Max Latency:                              {min(latencies):.2f} ms / {max(latencies):.2f} ms")
    print(f"Throughput:                                     {len(latencies)/(sum(latencies)/1000.0):.1f} items/sec ({len(latencies)*4/(sum(latencies)/1000.0):.1f} question decisions/sec)")
    print(f"Mean Volatility Severity Score:                 {sum(severity_scores)/len(severity_scores):.2f} / 3.00")
    print(f"Mean Regulatory Intervention Probability:       {sum(regulatory_probs)/len(regulatory_probs):.1%}")
    print("\nSector Distribution Breakdown:")
    for sec, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {sec:<35}: {count:2d} articles ({count/len(selected_indices):.1%})")
    print("=" * 100)

if __name__ == "__main__":
    run_real_finance_eval(100)

