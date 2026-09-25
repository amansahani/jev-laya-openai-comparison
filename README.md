# System 1 Decision Benchmarks: TypeSafe Jev vs. Laya vs. OpenAI

A reproducible benchmark suite and pure PyTorch / HuggingFace inference pipeline comparing **Non-Autoregressive System 1 Decision Models** ([TypeSafe Jev](https://docs.typesafe.ai) and [Laya](https://huggingface.co/convaiinnovations/laya)) against **Autoregressive Language Models** (OpenAI `gpt-5.6-luna` / `gpt-4o-mini`).

---

## Benchmark Summary (Indian Financial Regulatory Suite)

Evaluation conducted over 10 complex statutory scenarios (RBI Asset Classification, SEBI PIT Regulations, Cyber Financial Fraud, FEMA LRS limits, and Income Tax Section 194-IA):

| Model Architecture | Execution Mode | Choice Accuracy | Noul Brier Score (↓) | Score MAE (↓) | Avg Latency | Cost (10 Evals) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Laya (ModernBERT-421M)** | Local CUDA (`bfloat16`) | 60.0% | 0.5135 | 0.9305 | **99.7 ms** | **$0.000000** |
| **TypeSafe Jev (`jev-1.13.0`)** | Cloud REST API | **100.0%** | **0.0953** | 0.1120 | 927.1 ms | **$0.000239** |
| **OpenAI Luna (`gpt-5.6-luna`)** | Cloud REST API | **100.0%** | 0.0962 | **0.1000** | 3,547.6 ms | **$0.004490** |

*Pricing rates sourced directly from official documentation: TypeSafe Jev ($0.042/1M input, $0 output) vs. OpenAI ($0.20/1M input, $1.20/1M output).*

---

## Architecture Overview

```
Input Sequence Layout (Laya / Jev):
[CLS] choice question: Route inquiry [SEP] [MASK] opt_0 [MASK] opt_1 [MASK] opt_2 [SEP] State Context [SEP]
                                            ▲             ▲             ▲
                                            └── Gather 0  └── Gather 1  └── Gather 2 -> Softmax
```

- **Single-Pass Inference**: Evaluates all candidate choices simultaneously in ~25–50ms on GPU.
- **Zero Hallucination Risk**: Output spaces are bounded strictly to user-defined schemas.
- **Calibrated Probabilities**: Trained via Reinforcement Learning with Strictly Proper Scoring Rules (Brier / Ranked Probability Score).

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/amansahani/jev-laya-openai-comparison.git
cd jev-laya-openai-comparison
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

```env
JEV_API_KEY=your_typesafe_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LAYA_MODEL_PATH=./models/laya  # or your local checkpoint directory
```

### 3. Running Local Laya Pipeline (Pure PyTorch & Transformers)

```python
from pipeline import JevClassificationPipeline

pipeline = JevClassificationPipeline(model_path="./models/laya")

# 1. Zero-shot Classification (Choice)
result = pipeline.classify(
    text="Our credit card was charged twice for the monthly subscription.",
    labels={
        "billing": "invoices, payments, refunds, duplicate charges",
        "tech_support": "bugs, outages, system errors"
    },
    instruction="Which team should handle this inquiry?"
)
print("Label:", result["label"])
print("Probabilities:", result["probabilities"])

# 2. Boolean Claim Verification (Noul)
claim_result = pipeline.verify(
    text="Customer demands immediate contract termination by tomorrow.",
    claim="Does the customer threaten churn or cancellation?"
)
print("P(True):", claim_result["is_true_probability"])
```

### 4. Running the Benchmark Suite

```bash
# Run side-by-side evaluation across Local Laya, TypeSafe Jev, and OpenAI
python eval_indian_finance.py

# Run real-world evaluation on 100 Indian Financial News articles (HuggingFace)
python run_real_dataset_eval.py
```

---

## Repository Structure

```
.
├── pipeline.py                 # Standalone PyTorch/Transformers Laya inference engine
├── run_real_dataset_eval.py    # Real Indian Financial News dataset evaluation (HuggingFace)
├── eval_indian_finance.py       # Benchmark evaluation script across all 3 models
├── indian_finance_dataset.py   # Ground-truth test dataset of statutory financial scenarios
├── compare.py                  # Pairwise comparison runner (Laya vs Jev)
├── benchmark_all_three.py      # Tri-model concurrency test script
├── requirements.txt            # Python dependencies
└── .env.example                # Safe environment configuration template
```

---

## Citation

If you use this benchmark suite or inference implementation in your research, please cite:

```bibtex
@article{sahani2026system1benchmark,
  title={Stop Using 70B Language Models for If-Else Statements: A Comparative Benchmark of TypeSafe Jev, Laya, and OpenAI},
  author={Sahani, Aman and Systems Engineering Group},
  year={2026},
  url={https://github.com/amansahani/jev-laya-openai-comparison}
}
```
