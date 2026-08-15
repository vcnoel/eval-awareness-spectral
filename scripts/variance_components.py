"""Variance components and Clark's min-F' for the framing endpoint.

Two purposes.

1. Correct inference. Treating sampled framings as fixed effects is Clark's (1973)
   language-as-fixed-effect fallacy; Judd, Westfall & Kenny (2012) show it inflates
   significance, and Baayen, Davidson & Bates (2008) give the crossed random-effects fix.
   With one observation per (item, framing) cell we report Clark's F1 (items as the
   replication unit, framings fixed), F2 (framings as the unit, items fixed) and the
   conservative min-F', which is the classical solution when both factors are sampled.
   Bootstrapping six framings is NOT used here: percentile CIs at n=6 have poor coverage.

2. Sizing a matched-twin design. Pairing removes the part of the framing effect that is a
   constant offset across items, so its value depends on the item x family INTERACTION,
   not the family main effect. We decompose the matrix into item, family,
   framing-within-family, item x family, and residual components and report what a paired
   design would buy.

Caveat stated explicitly: with a single observation per (item, framing) cell, the residual
is confounded with the item x framing-within-family interaction. The item x FAMILY
interaction is identified, because each family contains six framings.

    python scripts/variance_components.py --matrix results/endpoint_gate_matrix.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# rendered prompt properties measured at build time (Qwen tokenizer), for the
# "what is predictable from surface form" regression
SURFACE = {
    "deploy_a": (143, 104), "eval_benchmark": (141, 102), "eval_redteam": (142, 103),
    "eval_certification": (130, 91), "eval_leaderboard": (131, 92), "eval_rubric": (128, 89),
    "eval_oversight": (128, 89), "neutral_documentation": (127, 88),
    "neutral_reference_library": (135, 96), "neutral_correspondence": (129, 90),
    "neutral_archive": (129, 90), "neutral_technical_support": (130, 91),
    "neutral_editorial": (132, 93),
}


def min_f_prime(f1: float, df1_error: int, f2: float, df2_error: int) -> tuple[float, float, float]:
    """Clark (1973) min-F' and its denominator degrees of freedom."""
    value = (f1 * f2) / (f1 + f2)
    df_denom = ((f1 + f2) ** 2) / ((f1 ** 2) / df2_error + (f2 ** 2) / df1_error)
    p = float(stats.f.sf(value, 1, df_denom))
    return value, df_denom, p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(ROOT / "results" / "endpoint_gate_matrix.json"))
    args = parser.parse_args()

    payload = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    names = payload["framings"]
    fam = payload["families"]
    matrix = np.asarray(payload["matrix_P_gold"], dtype=float)          # [items, framings]

    keep = [i for i, n in enumerate(names) if fam.get(n) in ("eval", "neutral")]
    y = matrix[:, keep]
    labels = [names[i] for i in keep]
    grp = np.array([0 if fam[n] == "eval" else 1 for n in labels])
    n_items, n_fr = y.shape
    print(f"matrix: {n_items} items x {n_fr} framings "
          f"({int((grp == 0).sum())} eval, {int((grp == 1).sum())} neutral)\n")

    # ---------------------------------------------------------------- variance components
    grand = y.mean()
    item_mean = y.mean(axis=1)
    fr_mean = y.mean(axis=0)
    fam_mean = np.array([y[:, grp == g].mean() for g in (0, 1)])
    # item x family cell means
    cell = np.stack([y[:, grp == g].mean(axis=1) for g in (0, 1)], axis=1)   # [items, 2]

    var_item = float(np.var(item_mean, ddof=1))
    # (no family "variance": over two group means it is just (difference^2)/2)
    var_fr_within = float(np.mean([np.var(fr_mean[grp == g], ddof=1) for g in (0, 1)]))
    inter = cell - item_mean[:, None] - fam_mean[None, :] + grand
    var_item_x_family = float(np.var(inter, ddof=1))
    fitted = item_mean[:, None] + fr_mean[None, :] - grand
    var_resid = float(np.var(y - fitted, ddof=1))

    print("=== VARIANCE COMPONENTS of P(gold) ===")
    for label, value in [
        ("item (task difficulty)", var_item),
        ("framing within family  <- pairing targets this", var_fr_within),
        ("item x family  <- paired design cannot remove this", var_item_x_family),
        ("residual (confounded with item x framing)", var_resid),
    ]:
        print(f"  {label:<48} {value:.5f}   sd={np.sqrt(value):.4f}")
    # NOTE: a "family variance" over two group means is just (difference^2)/2, i.e. a
    # rescaling of the effect itself, not a variance component. It is deliberately not
    # reported or compared against the nuisance components. The meaningful statement is
    # the one below: the effect is smaller than the between-framing nuisance sd.
    print(f"  [effect (eval-neutral) = {float(np.diff(fam_mean)[0] * -1):+.4f} vs "
          f"between-framing nuisance sd = {np.sqrt(var_fr_within):.4f}]")

    # ------------------------------------------------------------------- Clark's min-F'
    per_item = y[:, grp == 0].mean(axis=1) - y[:, grp == 1].mean(axis=1)
    t1 = stats.ttest_1samp(per_item, 0.0)
    f1, df1 = float(t1.statistic) ** 2, n_items - 1

    ev_means, nu_means = fr_mean[grp == 0], fr_mean[grp == 1]
    t2 = stats.ttest_ind(ev_means, nu_means, equal_var=True)
    f2, df2 = float(t2.statistic) ** 2, n_fr - 2

    mf, df_mf, p_mf = min_f_prime(f1, df1, f2, df2)
    diff = float(ev_means.mean() - nu_means.mean())

    print(f"\n=== FAMILY CONTRAST  (eval - neutral) = {diff:+.4f} ===")
    print(f"  F1 (items as unit, framings FIXED)   F={f1:8.3f} df=(1,{df1:>3})  p={t1.pvalue:.5f}"
          f"   <- the inflated test")
    print(f"  F2 (framings as unit, items fixed)   F={f2:8.3f} df=(1,{df2:>3})  p={t2.pvalue:.5f}")
    print(f"  Clark min-F'                         F={mf:8.3f} df=(1,{df_mf:5.1f})  p={p_mf:.5f}"
          f"   <- honest test")
    print(f"  inflation factor on F: {f1 / mf:.1f}x")

    # ------------------------------------------------- what a matched-twin design buys
    print("\n=== SIZING A MATCHED-TWIN DESIGN ===")
    sd_between = float(np.sqrt(var_fr_within))
    print(f"  unpaired between-framing sd (within family) = {sd_between:.4f}")
    print(f"  current unpaired SE at n=6/family           = {sd_between * np.sqrt(2 / 6):.4f}")
    print("  a twin pair removes the shared scenario offset; what remains is item x family")
    print(f"  interaction sd = {np.sqrt(var_item_x_family):.4f}")
    for n_pairs in (8, 12, 16, 20):
        # optimistic: pairing removes the framing-within-family component entirely
        se_paired = np.sqrt(var_item_x_family / n_pairs)
        print(f"    n={n_pairs:>2} twin pairs -> SE ~ {se_paired:.4f}, "
              f"detectable effect (2.1 SE) ~ {2.1 * se_paired:.4f}")
    print(f"  observed family difference to detect: {abs(diff):.4f}")

    # -------------------------------------------- what surface form already predicts
    print("\n=== FRAMING MEANS REGRESSED ON SURFACE PROPERTIES ===")
    rows = [(labels[i], fr_mean[i], *SURFACE[labels[i]]) for i in range(n_fr) if labels[i] in SURFACE]
    if rows:
        vals = np.array([r[1] for r in rows])
        toks = np.array([r[2] for r in rows], dtype=float)
        start = np.array([r[3] for r in rows], dtype=float)
        for nm, x in (("n_tokens", toks), ("task_start", start)):
            r, p = stats.pearsonr(x, vals)
            print(f"  {nm:<12} r={r:+.3f} p={p:.3f}  (R2={r ** 2:.3f})")
        design = np.column_stack([np.ones_like(toks), toks, start])
        beta, *_ = np.linalg.lstsq(design, vals, rcond=None)
        pred = design @ beta
        ss = 1 - ((vals - pred) ** 2).sum() / ((vals - vals.mean()) ** 2).sum()
        print(f"  joint R2 (length + position) = {ss:.3f}")
        print("\n  per-framing means, sorted (what a twin must match on):")
        for name, value, tk, st in sorted(rows, key=lambda r: r[1]):
            print(f"    {name:<28} {fam[name]:<8} P(gold)={value:.4f}  tok={tk:.0f} start={st:.0f}")


if __name__ == "__main__":
    main()
