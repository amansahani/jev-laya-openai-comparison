"""
Jev / Laya Classification Pipeline using pure PyTorch and HuggingFace Transformers.

Compatible with TypeSafe Jev decision schemas and Laya checkpoints.
Runs zero-shot typed classification (choice, score, noul) in a single forward pass.
"""

import os
import json
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Question type constants matching Jev / Laya specification
QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


class DecisionModel(nn.Module):
    """
    Pretrained bidirectional encoder backbone + decision head.
    Scores [MASK] markers corresponding to candidate options.
    """

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256),
            nn.GELU(),
            nn.Linear(256, n_act)
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                marker_pos: torch.Tensor, marker_mask: torch.Tensor, qtype: torch.Tensor):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            h = self.head(h, src_key_padding_mask=pad)

        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        # Act head (escalation / action probabilities)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def serialize_state(state: Any) -> str:
    """Serializes text, dict, or json-compatible objects to string."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(q: Dict[str, Any]) -> List[str]:
    """Generates human-readable option strings based on question definition."""
    t = q["t"]
    crit = q.get("crit")
    if t == "choice":
        if isinstance(crit, dict):
            return [k if not v else f"{k}: {v}" for k, v in crit.items()]
        elif isinstance(crit, list):
            return [str(c) for c in crit]
        return []
    if t == "score":
        if isinstance(crit, list):
            return [f"level {i}: {c}" for i, c in enumerate(crit)]
        elif isinstance(crit, dict):
            return [f"{k}: {v}" for k, v in crit.items()]
        return ["level 0: low", "level 1: medium", "level 2: high"]
    crit = crit or {}
    return [
        "false: " + (crit.get("false") or "no, the statement does not hold"),
        "true: " + (crit.get("true") or "yes, the statement holds")
    ]


def build_sequence(tok: AutoTokenizer, state: Any, q: Dict[str, Any], max_len: int, head_max_len: int):
    """
    Constructs model input sequence:
    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok(f"{q['t']} question: {ins}", add_special_tokens=False)["input_ids"]

    opt_ids = []
    for opt in opts:
        opt_ids.append([tok.mask_token_id] + tok(" " + opt.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48])

    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)

    head_ids = head_ids[:max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)

    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def temp_bucket(qtype: int, k: int) -> str:
    """Bucket key for cardinality-specific temperature scaling."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{QTYPE_NAMES[int(qtype)]}:{size}"


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Calibrated confidence: 1 - normalized entropy."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(1.0 - ent / math.log(k))


class JevClassificationPipeline:
    """
    High-performance, pure PyTorch & HuggingFace Transformers decision pipeline.
    Drop-in alternative to TypeSafe Jev, powered by local Laya weights.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        self.model_path = model_path or os.environ.get("LAYA_MODEL_PATH", "./models/laya")
        with open(os.path.join(self.model_path, "rl_agent_config.json"), "r") as f:
            self.cfg = json.load(f)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if dtype is None:
            if self.device.type == "cuda":
                self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        # Load Tokenizer & Encoder
        self.tok = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
        encoder_dir = os.path.join(model_path, "encoder")
        if os.path.exists(encoder_dir) and os.path.isdir(encoder_dir):
            encoder_cfg = AutoConfig.from_pretrained(encoder_dir)
            encoder = AutoModel.from_config(encoder_cfg, attn_implementation="sdpa")
        else:
            encoder = AutoModel.from_pretrained(self.cfg["encoder"], attn_implementation="sdpa")

        self.model = DecisionModel(
            encoder=encoder,
            head_layers=self.cfg.get("head_layers", 2),
            n_act=len(self.cfg.get("act_costs", {})) + 1
        )

        # Load Safetensors weights
        weights_path = os.path.join(model_path, "model.safetensors")
        weights = load_file(weights_path)
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval()

        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.max_len = self.cfg.get("max_len", 512)
        self.head_max_len = self.cfg.get("head_max_len", 192)

    @staticmethod
    def _normalize_question(qdef: Dict[str, Any]) -> Dict[str, Any]:
        t = qdef.get("t") or qdef.get("type", "choice")
        ins = qdef.get("ins") or qdef.get("question") or qdef.get("instructions", "")
        if not isinstance(ins, str):
            ins = json.dumps(ins)
            
        crit = qdef.get("crit") or qdef.get("criteria")
        if crit is None:
            if t == "choice":
                opts = qdef.get("options", [])
                if isinstance(opts, list):
                    crit = {c: None for c in opts}
                elif isinstance(opts, dict):
                    crit = opts
                else:
                    crit = {}
            elif t == "score":
                min_v = int(qdef.get("min", 0))
                max_v = int(qdef.get("max", 5))
                crit = [f"score {i}" for i in range(min_v, max_v + 1)]
            elif t == "noul":
                crit = {}
        elif t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        elif t == "score" and isinstance(crit, dict):
            crit = [crit.get(k, str(k)) for k in sorted(crit.keys())]

        return {"t": t, "ins": ins, "crit": crit}

    @torch.no_grad()
    def predict_questions(
        self,
        state: Any,
        questions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes multi-question Jev-style inference in a single forward pass.
        
        Args:
            state: Context string, conversation history, or metadata dict.
            questions: Dictionary mapping question IDs to question definitions.
                       e.g. {"intent": {"type": "choice", "instructions": "...", "criteria": {...}}}
        """
        qids = list(questions.keys())
        if not qids:
            return {"answers": {}, "model": "jev-laya"}

        items = []
        for qid in qids:
            q = self._normalize_question(questions[qid])
            seq, markers = build_sequence(self.tok, state, q, self.max_len, self.head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError(f"Question '{qid}' options do not fit in head_max_len={self.head_max_len}")
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

        # Collate batch
        n = len(items)
        max_seq_len = max(len(it["ids"]) for it in items)
        max_markers = max(len(it["markers"]) for it in items)

        input_ids = torch.full((n, max_seq_len), self.tok.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_seq_len), dtype=torch.long)
        marker_pos = torch.zeros((n, max_markers), dtype=torch.long)
        marker_mask = torch.zeros((n, max_markers), dtype=torch.bool)
        qtypes_tensor = torch.tensor([it["qtype"] for it in items], dtype=torch.long)

        for i, it in enumerate(items):
            input_ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
            attention_mask[i, :len(it["ids"])] = 1
            k = len(it["markers"])
            marker_pos[i, :k] = torch.tensor(it["markers"])
            marker_mask[i, :k] = True

        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=use_amp):
            logits, act_logits = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                marker_pos=marker_pos.to(self.device),
                marker_mask=marker_mask.to(self.device),
                qtype=qtypes_tensor.to(self.device)
            )

        logits = logits.float().cpu().numpy()
        act_probs = torch.softmax(act_logits.float(), -1).cpu().numpy()

        answers = {}
        for r, qid in enumerate(qids):
            q = self._normalize_question(questions[qid])
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            t_bucket = temp_bucket(qt, k)
            temp = self.temperature_by_options.get(t_bucket, self.temperature[qt])

            z = logits[r, :k] / temp
            p = np.exp(z - np.max(z))
            p = p / np.sum(p)

            act_prob = float(act_probs[r, 0])

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                top_idx = int(p.argmax())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[top_idx],
                    "prediction": keys[top_idx],
                    "probabilities": {k_name: round(float(v), 4) for k_name, v in zip(keys, p)},
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "act_probability": round(act_prob, 4)
                }
            elif q["t"] == "score":
                score_val = float(np.sum(np.arange(k) * p))
                answers[qid] = {
                    "type": "score",
                    "score": round(score_val, 4),
                    "prediction": round(score_val, 4),
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "act_probability": round(act_prob, 4)
                }
            else:  # noul (boolean)
                noul_prob = round(float(p[1]), 4)
                answers[qid] = {
                    "type": "noul",
                    "noul": noul_prob,
                    "probability": noul_prob,
                    "prediction": bool(noul_prob >= 0.5),
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "act_probability": round(act_prob, 4)
                }

        return {
            "model": "jev-laya",
            "answers": answers,
            "usage": {"input_tokens": int(attention_mask.sum()), "output_tokens": 0}
        }

    def classify(
        self,
        text: str,
        labels: Union[List[str], Dict[str, str]],
        instruction: str = "Select the most appropriate category."
    ) -> Dict[str, Any]:
        """
        Simple high-level classification API (zero-shot classification).
        
        Args:
            text: Input text to classify.
            labels: List of label strings OR Dict mapping label to description/criteria.
            instruction: Optional instruction guide for classification.
        """
        if isinstance(labels, list):
            crit = {lbl: None for lbl in labels}
        else:
            crit = labels

        res = self.predict_questions(
            state=text,
            questions={"classification": {"type": "choice", "instructions": instruction, "criteria": crit}}
        )
        ans = res["answers"]["classification"]
        return {
            "label": ans["choice"],
            "probabilities": ans["probabilities"],
            "confidence": ans["confidence"]
        }

    def verify(
        self,
        text: str,
        claim: str
    ) -> Dict[str, Any]:
        """
        Boolean verification / statement check (Noul type in Jev).
        Returns probability that the claim holds true.
        """
        res = self.predict_questions(
            state=text,
            questions={"verification": {"type": "noul", "instructions": claim}}
        )
        ans = res["answers"]["verification"]
        return {
            "is_true_probability": ans["noul"],
            "confidence": ans["confidence"]
        }

    def rate(
        self,
        text: str,
        levels: List[str],
        instruction: str = "Rate the intensity or level."
    ) -> Dict[str, Any]:
        """
        Ordinal scoring / severity rating (Score type in Jev).
        """
        res = self.predict_questions(
            state=text,
            questions={"rating": {"type": "score", "instructions": instruction, "criteria": levels}}
        )
        ans = res["answers"]["rating"]
        return {
            "score": ans["score"],
            "probabilities": ans["probabilities"],
            "confidence": ans["confidence"]
        }
