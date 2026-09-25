"""
Example demonstration of Jev / Laya Classification Pipeline using HuggingFace Transformers & PyTorch.
"""

import json
from pipeline import JevClassificationPipeline

def main():
    print("Loading Jev / Laya Classification Pipeline...")
    pipeline = JevClassificationPipeline()
    print(f"Model successfully loaded on device: {pipeline.device} (dtype: {pipeline.dtype})\n")

    # 1. Zero-shot Label Classification (Choice)
    sample_text_1 = "Hi, our credit card was charged twice for the enterprise subscription this month. Please issue a refund."
    labels = {
        "billing": "invoices, payments, refunds, and subscription charges",
        "technical_support": "bugs, service outages, system errors, and crashes",
        "account_management": "password resets, user permissions, and profile updates",
        "general_inquiry": "feature requests, sales queries, and general questions"
    }

    print("=" * 60)
    print("1. Multi-class Classification (Department Routing)")
    print(f"Input: \"{sample_text_1}\"")
    result_1 = pipeline.classify(
        text=sample_text_1,
        labels=labels,
        instruction="Which department should handle this customer support inquiry?"
    )
    print("Result:")
    print(json.dumps(result_1, indent=2))

    # 2. Boolean Verification / Threat & Churn Risk (Noul)
    sample_text_2 = "If you don't resolve this issue by tomorrow, we are canceling our contract and moving to a competitor."
    print("\n" + "=" * 60)
    print("2. Boolean Claim Verification / Risk Detection (Noul)")
    print(f"Input: \"{sample_text_2}\"")
    result_2 = pipeline.verify(
        text=sample_text_2,
        claim="Does the customer threaten to cancel their account or leave?"
    )
    print("Result:")
    print(json.dumps(result_2, indent=2))

    # 3. Ordinal Rating / Urgency Scoring (Score)
    sample_text_3 = "The entire production database is down and all our users cannot log in!"
    levels = ["low: standard request, no rush", "medium: inconvenience, needs attention soon", "high: blocking issue affecting key operations", "critical: complete production outage"]
    print("\n" + "=" * 60)
    print("3. Ordinal Severity / Urgency Scoring")
    print(f"Input: \"{sample_text_3}\"")
    result_3 = pipeline.rate(
        text=sample_text_3,
        levels=levels,
        instruction="How severe or urgent is this incident?"
    )
    print("Result:")
    print(json.dumps(result_3, indent=2))

    # 4. Multi-Question Simultaneous Forward Pass (Jev / TypeSafe Schema)
    sample_text_4 = "I found a critical vulnerability in your authentication endpoint. We need this escalated immediately."
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Route this ticket to the right team",
            "criteria": {
                "security": "vulnerabilities, exploits, CVEs, auth bypass",
                "billing": "pricing, invoices, payment methods",
                "infra": "server maintenance, network configuration"
            }
        },
        "urgency": {
            "type": "score",
            "instructions": "Assess the urgency level",
            "criteria": ["low", "medium", "high", "critical"]
        },
        "is_security_exploit": {
            "type": "noul",
            "instructions": "Does this report discuss a security vulnerability or exploit?"
        }
    }

    print("\n" + "=" * 60)
    print("4. Multi-Question System 1 Single Forward Pass (Full Jev Schema)")
    print(f"Input: \"{sample_text_4}\"")
    result_4 = pipeline.predict_questions(state=sample_text_4, questions=questions)
    print("Result:")
    print(json.dumps(result_4, indent=2))


if __name__ == "__main__":
    main()
