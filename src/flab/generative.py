"""Generative exact-match evaluation (phase-1b open item 14).

The calibration replicated 2606.27634's drift ordering exactly but not their
KL→accuracy link, and the whole disagreement is one swap: their gemma scores
worst (0.320), ours scores mid-field (0.586). The leading suspect is that **our
accuracy is teacher-forced and theirs is generative** — a model that would
ramble in free generation still scores well when the answer tokens are handed
to it. This module is what decides that, and nothing else can.

The normalisation reproduces their `src/utils.py` (HEAD 801a3b3): strip
generation artefacts, then extract an option letter for multiple-choice tasks
and a canonical number for numeric ones, preferring explicit answer markers and
falling back to the last candidate in the text.

Note their normaliser has **no `\\boxed{}` handling** despite the NumGLUE prompt
asking for it — `extract_number` catches it only incidentally, via the
last-number fallback. Reproduced as-is rather than improved: a "better"
normaliser would not be a replication.

**Adapters are merged before generating.** Measured 2026-08-07: an unmerged
adapter costs 1.89× on generative decode, and this is the expensive eval.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

MULTIPLE_CHOICE_TASKS = ("ScienceQA", "FOMC")
NUMERIC_TASKS = ("NumGLUE-cm",)


def strip_generation_artifacts(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for marker in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>"):
        text = text.replace(marker, "").strip()
    return text.replace("**", "").replace("__", "").strip()


def normalize_number(text: str) -> str:
    text = re.sub(r"[.$,;:]+$", "", text.strip().replace(",", ""))
    try:
        value = float(text)
        return str(int(value)) if value.is_integer() else str(value)
    except ValueError:
        return text


def extract_multiple_choice(text: str) -> str:
    text = strip_generation_artifacts(text)
    for pattern in (r"final answer\s*[:\-]\s*([A-E])\b", r"answer\s*[:\-]\s*([A-E])\b",
                    r"the answer is\s*([A-E])\b", r"option\s*([A-E])\b",
                    r"choice\s*([A-E])\b"):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.match(r"^\s*([A-E])(?:[.)]|\s|$)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b([A-E])\b", text)
    return m[-1].upper() if m else text.strip().upper()


def extract_number(text: str) -> str:
    text = strip_generation_artifacts(text)
    for pattern in (r"final answer\s*[:\-]\s*(.*)", r"answer\s*[:\-]\s*(.*)",
                    r"the answer is\s*(.*)", r"therefore,?\s*(.*)"):
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", m.group(1))
            if nums:
                return normalize_number(nums[-1])
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return normalize_number(nums[-1]) if nums else text.strip().upper()


def normalize_answer(text: str, task: str | None = None) -> str:
    text = strip_generation_artifacts(text)
    if task in MULTIPLE_CHOICE_TASKS:
        return extract_multiple_choice(text)
    if task in NUMERIC_TASKS:
        return extract_number(text)
    m = re.search(r"final answer\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).splitlines()[0].strip().upper()
    lines = text.splitlines()
    return lines[-1].strip().upper() if lines else text.upper()


@dataclass
class GenResult:
    task: str
    exact_match: float
    n: int
    seconds: float
    examples: list[dict]     # a few (gold, pred, raw) for eyeballing


def evaluate(model, tokenizer, task: str, examples: list[dict], *,
             max_new_tokens: int = 256, batch_size: int = 8,
             prompt_style: str = "paper", keep: int = 5) -> GenResult:
    """Greedy generation, then normalized exact match — their `evaluate_accuracy`."""
    import time

    import torch

    from flab import prompts

    t0 = time.perf_counter()
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"      # generation needs left padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct, shown = 0, []
    was_training = model.training
    model.eval()
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        texts = [prompts.render(prompt_style, task, e["prompt"], e["answer"], tokenizer)[0]
                 for e in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 num_beams=1, pad_token_id=tokenizer.pad_token_id)
        for r, e in enumerate(chunk):
            raw = tokenizer.decode(out[r, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            gold, pred = normalize_answer(e["answer"], task), normalize_answer(raw, task)
            correct += gold == pred
            if len(shown) < keep:
                shown.append({"gold": gold, "pred": pred, "raw": raw[:120]})
    if was_training:
        model.train()

    return GenResult(task=task, exact_match=correct / len(examples), n=len(examples),
                     seconds=time.perf_counter() - t0, examples=shown)
