"""
Tri-Model Benchmark: Local Laya vs Cloud TypeSafe Jev vs OpenAI Luna (gpt-5.6-luna).
"""

import os
import time
import json
import requests
from typing import Dict, Any
from pipeline import JevClassificationPipeline

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

def query_laya(pipeline: JevClassificationPipeline, state: str, questions: Dict[str, Any]):
    t0 = time.perf_counter()
    res = pipeline.predict_questions(state, questions)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    res["_latency_ms"] = latency_ms
    return res

def query_jev(api_key: str, state: str, questions: Dict[str, Any]):
    url = "https://api.typesafe.ai/v1/systemone"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"state": state, "model": "jev-latest", "questions": questions}
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    data = resp.json()
    data["_latency_ms"] = latency_ms
    return data

def query_openai_luna(api_key: str, state: str, questions: Dict[str, Any]):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    prompt = f"""Evaluate the following text and answer the questions in strict JSON format.

Text:
"{state}"

Questions to answer:
{json.dumps(questions, indent=2)}

Return a JSON object with format:
{{
  "answers": {{
    "<question_id>": {{
      "choice": "<selected_choice_if_choice_type>",
      "score": <float_score_if_score_type>,
      "is_true": <boolean_if_noul_type>,
      "confidence": <float_0_to_1>
    }}
  }}
}}"""

    payload = {
        "model": "gpt-5.6-luna",
        "messages": [
            {"role": "system", "content": "You are a precise classifier and decision engine. Always output pure valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    data = resp.json()
    try:
        content = json.loads(data["choices"][0]["message"]["content"])
    except Exception:
        content = {"raw": data["choices"][0]["message"]["content"]}
    return {
        "model": data.get("model", "gpt-5.6-luna"),
        "answers": content.get("answers", content),
        "usage": data.get("usage", {}),
        "_latency_ms": latency_ms
    }

def main():
    env = load_env(".env")
    jev_key = env.get("JEV_API_KEY")
    openai_key = env.get("OPENAI_API_KEY")

    print(f"JEV API Key configured: {'Yes' if jev_key else 'No'}")
    print(f"OpenAI API Key configured: {'Yes' if openai_key else 'No'}")

    model_path = env.get("LAYA_MODEL_PATH", "./models/laya")
    laya = JevClassificationPipeline(model_path=model_path)

    state = "Our team discovered a critical unauthenticated remote code execution (RCE) flaw in your auth microservice. We demand an immediate hotfix or we will publish a full zero-day disclosure."
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Route to the appropriate response team",
            "criteria": {
                "security": "CVEs, RCEs, zero-days, exploit triage",
                "billing": "invoices, payments, refunds",
                "customer_support": "general user queries"
            }
        },
        "is_threat_or_exploit": {
            "type": "noul",
            "instructions": "Does the message contain a critical security threat or exploit disclosure?"
        },
        "urgency": {
            "type": "score",
            "instructions": "Rate urgency from 0 (low) to 3 (emergency blocker)",
            "criteria": ["routine", "moderate", "high", "critical emergency"]
        }
    }

    print("\n" + "="*80)
    print("RUNNING TRI-MODEL EVALUATION")
    print("="*80)
    print(f"Input State: \"{state}\"\n")

    # 1. Laya
    print("Executing Local Laya...")
    laya_res = query_laya(laya, state, questions)
    print(f" [Laya] Latency: {laya_res['_latency_ms']:.1f} ms")
    print(json.dumps(laya_res["answers"], indent=2))

    # 2. TypeSafe Jev
    print("\nExecuting TypeSafe Jev...")
    jev_res = query_jev(jev_key, state, questions)
    print(f" [TypeSafe Jev] Latency: {jev_res['_latency_ms']:.1f} ms")
    print(json.dumps(jev_res["answers"], indent=2))

    # 3. OpenAI Luna
    print("\nExecuting OpenAI Luna (gpt-5.6-luna)...")
    luna_res = query_openai_luna(openai_key, state, questions)
    print(f" [OpenAI Luna] Latency: {luna_res['_latency_ms']:.1f} ms")
    print(json.dumps(luna_res["answers"], indent=2))

if __name__ == "__main__":
    main()
