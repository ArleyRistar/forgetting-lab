"""Synthetic forgetting controls with analytically known answers (1b task 5).

Spec §6 1b asks for this alongside the replication, and it does different work.
The replication says "we agree with the published literature". The control says
"our instrument reads correctly on a case where the answer is known before the
run starts" — which is the only kind of evidence that catches an instrument
that is self-consistently wrong.

Two paired tasks, both key→value recall over nonsense keys:

**conflict** — A maps `k → v₁`, B maps *the same* `k → v₂`. Learning B to
convergence **must** destroy A; not "probably", but because one distribution
cannot concentrate on two different values of the same key. If the model puts
mass p on v₂ then at most 1−p remains for v₁, so A's NLL is bounded below by
−log(1−p) and diverges as p → 1. **Large forgetting is the known answer.**

**disjoint** — A maps `k_A → v_A`, B maps a disjoint `k_B → v_B`. Nothing
forces interference; a model with spare capacity can hold both. **Zero
forgetting is the known answer**, and whatever we actually measure is the
harness's *noise floor* — the level below which phase 2 must not report effects.

Values are drawn from a fixed set of N_VALUES options, which puts an analytic
scale on the NLL:

    ~0            perfect recall
    log(8) ≈ 2.08 chance, i.e. the association is gone
    ≫ log(8)      confidently wrong — the conflicting value was learned instead
"""
from __future__ import annotations

import hashlib

# Single-character values, so one answer token carries the whole association and
# the analytic chance level is exactly log(N_VALUES).
VALUES = ["A", "B", "C", "D", "E", "F", "G", "H"]
N_VALUES = len(VALUES)

PAIRS = ("conflict", "disjoint")
TASKS = tuple(f"synth-{p}-{s}" for p in PAIRS for s in ("a", "b"))


def chance_nll() -> float:
    """NLL of a uniform guess over the value set — the 'association is gone' mark."""
    import math

    return math.log(N_VALUES)


def _rand(*parts: object) -> int:
    """Deterministic pseudo-random integer from the parts, stable across runs."""
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _key(namespace: str, i: int) -> str:
    """A nonsense key. Namespaced so `disjoint` really is disjoint."""
    h = hashlib.sha256(f"{namespace}:{i}".encode()).hexdigest()
    return f"{h[:4]}-{h[4:8]}"


def make(task: str, split: str, n_keys: int = 50, repeats: int = 4,
         seed: int = 0) -> list[dict]:
    """Generate one synthetic task split as `{prompt, answer}` rows.

    `train` repeats each key so the association is actually memorised; `eval`
    holds each key once. Train and eval deliberately cover the **same** keys:
    this measures recall of what was taught, which is what forgetting means
    here, not generalisation to unseen keys.
    """
    if task not in TASKS:
        raise ValueError(f"unknown synthetic task {task!r}; known: {TASKS}")
    pair, side = task.rsplit("-", 2)[-2], task.rsplit("-", 1)[-1]

    # conflict: both sides share a key namespace, so B overwrites A's keys.
    # disjoint: each side gets its own namespace, so nothing collides.
    namespace = pair if pair == "conflict" else f"{pair}-{side}"

    rows = []
    for i in range(n_keys):
        k = _key(namespace, i)
        # The value depends on the *side*, so conflict-a and conflict-b assign
        # different values to the identical key — which is the whole point.
        v = VALUES[_rand(seed, side, namespace, i) % N_VALUES]
        if pair == "conflict" and side == "b":
            # Force a genuine conflict: never accidentally agree with A.
            va = VALUES[_rand(seed, "a", namespace, i) % N_VALUES]
            if v == va:
                v = VALUES[(VALUES.index(v) + 1) % N_VALUES]
        rows.extend([{
            "prompt": f"The key {k} maps to which value? Answer with one letter.",
            "answer": v,
        }] * (repeats if split == "train" else 1))
    return rows


def conflicting_values(n_keys: int = 50, seed: int = 0) -> int:
    """How many keys genuinely differ between conflict-a and conflict-b.

    Should be every one of them — a control that silently agreed on some keys
    would understate the forgetting it is supposed to guarantee.
    """
    a = {r["prompt"]: r["answer"] for r in make("synth-conflict-a", "eval", n_keys, seed=seed)}
    b = {r["prompt"]: r["answer"] for r in make("synth-conflict-b", "eval", n_keys, seed=seed)}
    return sum(1 for k in a if a[k] != b.get(k))
