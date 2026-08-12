"""Tests for the phase-2b fixed-effects estimator.

The central test plants a known gamma in synthetic log-probabilities and checks
it is recovered. An estimator that cannot recover a planted effect is not
evidence about anything, and this project has shipped two instruments that
measured something other than their name.
"""
import numpy as np
import pytest

from flab import fe

LETTERS = list("ABCDEFGH")


def _synthetic(gamma_v1: float, gamma_vp: float, n_keys: int = 200, seed: int = 0):
    """Log-probs built from a known key effect, letter effect and gamma."""
    rng = np.random.default_rng(seed)
    key_eff = {f"k{i}": rng.normal(-8, 1.5) for i in range(n_keys)}
    letter_eff = {l: rng.normal(0, 0.8) for l in LETTERS}
    v1, vp, v2, logp = {}, {}, {}, {}
    for k in key_eff:
        a, b, c = rng.choice(LETTERS, size=3, replace=False)
        v1[k], v2[k], vp[k] = a, b, c
        for l in LETTERS:
            if l == v2[k]:
                continue
            val = key_eff[k] + letter_eff[l] + rng.normal(0, 0.05)
            if l == v1[k]:
                val += gamma_v1
            if l == vp[k]:
                val += gamma_vp
            logp[(k, l)] = val
    return logp, v1, vp, v2


def test_recovers_a_planted_gamma():
    logp, v1, vp, v2 = _synthetic(gamma_v1=0.65, gamma_vp=0.0)
    r = fe.fit(logp, v1, vp, v2)
    assert r.gamma_v1 == pytest.approx(0.65, abs=0.05)
    assert r.gamma_vp == pytest.approx(0.0, abs=0.05)


def test_recovers_zero_when_there_is_no_trace():
    """The null the placebo gate depends on."""
    logp, v1, vp, v2 = _synthetic(gamma_v1=0.0, gamma_vp=0.0, seed=3)
    r = fe.fit(logp, v1, vp, v2)
    assert abs(r.gamma_v1) < 0.05


def test_separates_the_two_dummies():
    logp, v1, vp, v2 = _synthetic(gamma_v1=0.6, gamma_vp=-0.4, seed=5)
    r = fe.fit(logp, v1, vp, v2)
    assert r.gamma_v1 == pytest.approx(0.6, abs=0.06)
    assert r.gamma_vp == pytest.approx(-0.4, abs=0.06)


def test_letter_marginals_do_not_leak_into_gamma():
    """The failure the raw v1/distractor ratio had: one letter far more likely
    overall must not read as a trace."""
    logp, v1, vp, v2 = _synthetic(gamma_v1=0.0, gamma_vp=0.0, seed=7)
    for (k, l) in list(logp):
        if l == "D":
            logp[(k, l)] += 3.0          # huge letter marginal
    r = fe.fit(logp, v1, vp, v2)
    assert abs(r.gamma_v1) < 0.05, "letter effect leaked into gamma"


def test_key_effect_absorbs_b_mastery_differences():
    """A key whose total non-v2 mass is shifted must not move gamma — this is
    the mechanism behind the retracted 2-of-6-nats artefact."""
    logp, v1, vp, v2 = _synthetic(gamma_v1=0.5, gamma_vp=0.0, seed=11)
    shifted = {kk: val + (2.5 if int(kk[0][1:]) % 2 == 0 else 0.0)
               for kk, val in ((k, v) for k, v in logp.items())}
    r0 = fe.fit(logp, v1, vp, v2)
    r1 = fe.fit(shifted, v1, vp, v2)
    assert r1.gamma_v1 == pytest.approx(r0.gamma_v1, abs=0.02)


def test_paired_contrast_recovers_the_difference():
    logp_a, v1, vp, v2 = _synthetic(gamma_v1=0.65, gamma_vp=0.0, seed=13)
    logp_p, _, _, _ = _synthetic(gamma_v1=0.0, gamma_vp=0.0, seed=13)
    per_key = fe.paired_contrast(logp_a, logp_p, v1, v2)
    assert len(per_key) > 150
    assert float(np.mean(list(per_key.values()))) == pytest.approx(0.65, abs=0.08)


def test_seed_level_uses_t_with_two_df():
    r = fe.seed_level([0.6, 0.7, 0.65])
    assert r["n_seeds"] == 3 and r["crit_t"] == 4.303
    assert r["ci95_hi"] - r["ci95_lo"] > 2 * 1.96 * r["se"], "must be wider than z"


def test_seed_level_null_straddles_zero():
    r = fe.seed_level([0.1, -0.2, 0.05])
    assert r["ci95_lo"] < 0 < r["ci95_hi"]


def test_paired_contrast_is_bias_corrected():
    """Without the n/(n-1) correction a planted 0.65 returns 0.557: the v1 cell
    is inside the key mean it is measured against."""
    logp_a, v1, vp, v2 = _synthetic(gamma_v1=1.0, gamma_vp=0.0, seed=17)
    logp_p, _, _, _ = _synthetic(gamma_v1=0.0, gamma_vp=0.0, seed=17)
    m = float(np.mean(list(fe.paired_contrast(logp_a, logp_p, v1, v2).values())))
    assert m == pytest.approx(1.0, abs=0.05), f"got {m}, uncorrected would be ~0.857"


def test_placebo_letter_leaks_into_the_contrast_unless_excluded():
    """The real placebo condition TEACHES v', so v' carries its own trace there.
    Leaving it in the key mean leaks gamma_vp/6 into the contrast — 13-15% of the
    headline on real data. The earlier planted test missed this by planting
    gamma_vp = 0 in both conditions, the one case where the leak vanishes."""
    logp_a, v1, vp, v2 = _synthetic(gamma_v1=0.6, gamma_vp=0.0, seed=23)
    logp_p, _, _, _ = _synthetic(gamma_v1=0.0, gamma_vp=0.9, seed=23)  # v' taught

    leaky = float(np.mean(list(fe.paired_contrast(logp_a, logp_p, v1, v2).values())))
    clean = float(np.mean(list(
        fe.paired_contrast(logp_a, logp_p, v1, v2, vp=vp).values())))
    assert clean == pytest.approx(0.6, abs=0.06), f"clean estimate off: {clean}"
    assert leaky > clean + 0.08, (
        f"leak should inflate: leaky={leaky:.3f} clean={clean:.3f}")
