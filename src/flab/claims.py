"""Guards that stop an NLL-only claim from being reported as a capability claim.

**Why this module exists.** On 2026-08-11 this project reported "the ternary model
forgets less" from a ~6-nat NLL gap, and had to retract it: task-A token accuracy
was **0.00 in both arms**, so both models had forgotten completely and the entire
gap lived in the log-probability of a token neither model ever predicted. The
accuracy was sitting in every probe file for all 72 runs and was never read.

That was the fifth instance of the same shape — the turn-terminator artefact
(2026-08-09) was identical, and was also caught by looking at an accuracy that had
been in the output all along. Adversarial review caught three of the five. Review
is luck; a guard in the analysis path is not.

Two rules, both cheap:

1. **An NLL far above a task's chance level is outside its useful range.** Once
   the true answer's probability is ~1e-4, differences between models are
   comparisons of softmax tail mass, not of what either model would ever do.
2. **A forgetting claim needs a behavioural witness.** If accuracy is at floor in
   every arm, there is no retention to compare and an NLL difference cannot be
   described as one model remembering more.
"""
from __future__ import annotations

from dataclasses import dataclass

# How far above chance an NLL may sit before differences stop being meaningful.
# 2 nats ~ the true answer at e^-2 of chance probability; beyond that the metric
# is ranking tail mass. Deliberately generous — the retracted claim sat 7-14
# nats above chance.
NLL_HEADROOM_NATS = 2.0


@dataclass
class ClaimCheck:
    ok: bool
    reasons: list[str]

    def require(self) -> None:
        if not self.ok:
            raise AssertionError(
                "refusing to report a capability claim:\n  - "
                + "\n  - ".join(self.reasons))


def check_forgetting_claim(*, nlls: dict[str, float],
                           accuracies: dict[str, float | None],
                           chance_nll: float,
                           accuracy_floor: float = 0.0,
                           tolerance: float = 1e-9) -> ClaimCheck:
    """Is a between-arm NLL difference reportable as a difference in retention?

    `nlls` and `accuracies` are keyed by arm. `chance_nll` is the task's analytic
    chance level (`synthetic.chance_nll()` for the synth sets, `log(n_classes)`
    generally). `accuracy_floor` is the accuracy a model gets by guessing.

    Returns the reasons rather than raising, so a caller can report them.
    """
    reasons: list[str] = []

    known = {a: v for a, v in accuracies.items() if v is not None}
    if not known:
        reasons.append(
            "no accuracy recorded for any arm — an NLL difference alone cannot "
            "distinguish 'remembers more' from 'is differently calibrated'")
    elif all(v <= accuracy_floor + tolerance for v in known.values()):
        reasons.append(
            f"accuracy is at floor ({accuracy_floor}) in every arm "
            f"({known}) — both models have forgotten completely, so the NLL gap "
            "is tail mass, not retention")

    far = {a: v for a, v in nlls.items() if v > chance_nll + NLL_HEADROOM_NATS}
    if far:
        reasons.append(
            f"NLL is >{NLL_HEADROOM_NATS} nats above chance ({chance_nll:.4f}) "
            f"for {far} — differences up here compare softmax tails, and a model "
            "at 7 nats above chance is not 'remembering less', it is confidently "
            "wrong")

    return ClaimCheck(ok=not reasons, reasons=reasons)


def check_matched_capability(values: dict[str, float], *, label: str,
                             rel_tolerance: float = 0.5) -> ClaimCheck:
    """Are two arms close enough on a quantity to call it 'matched'?

    Compares on the RATIO, not the absolute difference. The retracted claim
    described B-mastery NLLs of 0.000532 and 0.000075 as identical because both
    round to ~0.000; they differ by 7x, and since A's NLL is bounded below by
    -log(1-p(B)), that gap mechanically produced ~2 of the 6 nats claimed as
    retention.
    """
    v = {k: abs(x) for k, x in values.items()}
    lo, hi = min(v.values()), max(v.values())
    if lo <= 0:
        return ClaimCheck(True, [])
    ratio = hi / lo
    if ratio > 1 + rel_tolerance:
        return ClaimCheck(False, [
            f"{label} differs by {ratio:.1f}x across arms ({values}) — 'matched' "
            "needs a ratio, not two numbers that both round to zero"])
    return ClaimCheck(True, [])
