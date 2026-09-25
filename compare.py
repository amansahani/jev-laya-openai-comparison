"""
Comprehensive Comparison Benchmark: Local Laya (PyTorch/Transformers) vs Cloud TypeSafe Jev API.
"""

import os
import time
import json
import requests
from typing import Dict, Any, List
from pipeline import JevClassificationPipeline

def load_typesafe_api_key(env_path: str = ".env") -> str:
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Could not find {env_path}")
    with open(env_path) as f:
        content = f.read().strip()
    if "=" in content:
        return content.split("=", 1)[1].strip()
    return content.strip()

def query_typesafe_api(api_key: str, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
    url = "https://api.typesafe.ai/v1/systemone"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "state": state,
        "model": "jev-latest",
        "questions": questions
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code != 200:
        raise RuntimeError(f"TypeSafe API error {resp.status_code}: {resp.text}")
    data = resp.json()
    data["_latency_ms"] = latency_ms
    return data

def run_comparison():
    api_key = load_typesafe_api_key(".env")
    print(f"Loaded TypeSafe API Key: {api_key[:12]}...{api_key[-6:]}")

    print("\nInitializing Local Laya Pipeline...")
    model_path = os.environ.get("LAYA_MODEL_PATH", "./models/laya")
    laya_pipeline = JevClassificationPipeline(model_path=model_path)
    print(f"Local Laya loaded on {laya_pipeline.device} ({laya_pipeline.dtype})\n")

    # Benchmark Test Suite
    test_cases = [
        {
            "name": "Case 1: Billing & Refund Request",
            "state": "Hi team, we were accidentally charged twice on invoice #9042 for our monthly pro plan. Please process a refund immediately.",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Route this inquiry to the most relevant team",
                    "criteria": {
                        "billing": "invoices, credit card charges, refunds, subscription plans",
                        "tech_support": "product bugs, error codes, crash logs",
                        "sales": "upgrades, annual contracts, enterprise quotes",
                        "security": "account compromise, suspicious login, vulnerability"
                    }
                },
                "churn_risk": {
                    "type": "noul",
                    "instructions": "Does the customer threaten to cancel their account or leave?"
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate how urgent this ticket is",
                    "criteria": ["low: minor question", "medium: standard request", "high: financial or operational impact", "critical: blocker"]
                }
            }
        },
        {
            "name": "Case 2: Threatening Churn & Contract Cancellation",
            "state": "Your platform has been down 3 times this week. If this isn't resolved by 5 PM today, our CEO will terminate our annual agreement and dispute the payment.",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which department should handle this escalation?",
                    "criteria": {
                        "billing": "payments and refunds",
                        "customer_success": "escalations, relationship management, retention",
                        "technical": "infrastructure, outages, bug fixes",
                        "sales": "new sales"
                    }
                },
                "churn_threat": {
                    "type": "noul",
                    "instructions": "Is the customer explicitly threatening churn or contract termination?"
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate the urgency of this escalation",
                    "criteria": ["routine", "moderate", "urgent", "critical emergency"]
                }
            }
        },
        {
            "name": "Case 3: Security & Exploit Disclosure",
            "state": "I discovered an unauthenticated SSRF vulnerability in your webhook processing service that leaks AWS metadata credentials.",
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Categorize the nature of this report",
                    "criteria": {
                        "security_vulnerability": "exploits, CVEs, credential leaks, auth bypass",
                        "feature_request": "asking for new product capabilities",
                        "bug_report": "general functional defect",
                        "general_question": "inquiries and docs"
                    }
                },
                "is_critical_security": {
                    "type": "noul",
                    "instructions": "Does this report present an active security threat or credential leak?"
                },
                "severity": {
                    "type": "score",
                    "instructions": "What is the severity of this issue?",
                    "criteria": ["informational", "low", "medium", "high", "critical"]
                }
            }
        },
        {
            "name": "Case 4: Ambiguous / Feature Question",
            "state": "Does your REST API support Webhooks for user signup events in addition to payment webhooks?",
            "questions": {
                "topic": {
                    "type": "choice",
                    "instructions": "What is the primary topic of this inquiry?",
                    "criteria": {
                        "api_documentation": "questions about endpoints, webhooks, parameters, SDKs",
                        "billing": "pricing, fees, invoices",
                        "bug": "defect or broken behavior"
                    }
                },
                "is_complaint": {
                    "type": "noul",
                    "instructions": "Is the user complaining or expressing dissatisfaction?"
                }
            }
        }
    ]

    print("=" * 80)
    print(f"{'BENCHMARK RESULTS: LOCAL LAYA vs CLOUD TYPESAFE JEV':^80}")
    print("=" * 80)

    laya_times = []
    jev_times = []

    for idx, case in enumerate(test_cases, 1):
        print(f"\n[{case['name']}]")
        print(f"Input State: \"{case['state'][:90]}...\"")
        
        # 1. Warm-up & Run Local Laya
        t0 = time.perf_counter()
        laya_res = laya_pipeline.predict_questions(case["state"], case["questions"])
        laya_latency = (time.perf_counter() - t0) * 1000.0
        laya_times.append(laya_latency)

        # 2. Run Cloud TypeSafe Jev
        jev_res = query_typesafe_api(api_key, case["state"], case["questions"])
        jev_latency = jev_res["_latency_ms"]
        jev_times.append(jev_latency)

        print(f"-> Latency: Local Laya = {laya_latency:.1f} ms | Cloud Jev = {jev_latency:.1f} ms")
        print("-" * 80)

        # Compare Question by Question
        for qid in case["questions"]:
            qtype = case["questions"][qid]["type"]
            l_ans = laya_res["answers"].get(qid, {})
            j_ans = jev_res["answers"].get(qid, {})

            print(f" Question: '{qid}' (Type: {qtype})")
            if qtype == "choice":
                l_choice = l_ans.get("choice")
                j_choice = j_ans.get("choice")
                l_top_prob = l_ans.get("probabilities", {}).get(l_choice, 0.0)
                j_top_prob = j_ans.get("probabilities", {}).get(j_choice, 0.0)
                match_str = "MATCH" if l_choice == j_choice else "DIFFER"
                print(f"   [Laya] -> Choice: '{l_choice}' (P={l_top_prob:.2f}, Conf={l_ans.get('confidence', 0):.2f})")
                print(f"   [Jev ] -> Choice: '{j_choice}' (P={j_top_prob:.2f}, Conf={j_ans.get('confidence', 0):.2f})  [{match_str}]")
            elif qtype == "noul":
                l_p = l_ans.get("noul", 0.0)
                j_p = j_ans.get("noul", 0.0)
                print(f"   [Laya] -> P(True) = {l_p:.3f} (Conf={l_ans.get('confidence', 0):.2f})")
                print(f"   [Jev ] -> P(True) = {j_p:.3f}")
            elif qtype == "score":
                l_s = l_ans.get("score", 0.0)
                j_s = j_ans.get("score", 0.0)
                print(f"   [Laya] -> Score = {l_s:.2f} (Conf={l_ans.get('confidence', 0):.2f})")
                print(f"   [Jev ] -> Score = {j_s:.2f} (Conf={j_ans.get('confidence', 0):.2f})")

    avg_laya = sum(laya_times) / len(laya_times)
    avg_jev = sum(jev_times) / len(jev_times)

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"Average Latency: Local Laya (GPU) = {avg_laya:.1f} ms  |  Cloud Jev API = {avg_jev:.1f} ms  (~{avg_jev/avg_laya:.1f}x speedup locally)")
    print(f"Cloud Jev Model Version: {jev_res.get('model')}")
    print(f"Local Backbone: {laya_pipeline.cfg.get('encoder')} ({laya_pipeline.cfg.get('model_name')})")

if __name__ == "__main__":
    run_comparison()
