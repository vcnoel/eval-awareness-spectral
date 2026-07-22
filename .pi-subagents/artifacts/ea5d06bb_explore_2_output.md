I don't have a file-write tool available in this session (only read/find/grep/cluster/artifact-store readers), so I cannot create the file at the required output path. I'm returning the full audit as my final message instead.

# Audit: vcnoel/eval-awareness-spectral @ 4bfd36c39494

## 1. Manuscript/venue mismatch (highest-severity documentation defect)
- **main.tex** header comment says `%%%%%%%% ICML 2026 SUBMISSION FILE %%%%%%%%%%%%%%%%%`, uses `\usepackage[accepted]{icml2026}`, `\icmlauthorlist`, `\bibliographystyle{icml2026}` — this is an **ICML** template, not ICLR.
- **docs/experiment_design.md** header: `# Experimental design: eval-awareness spectral fingerprint (ICLR-track target)`.
- Task brief calls for an "ICLR outline matching the accepted plan" but the actual LaTeX is ICML-formatted. This is an unresolved venue inconsistency across `main.tex` vs `docs/experiment_design.md` — must be reconciled before submission (either retarget the .tex to iclr2026 class or fix the doc header).
- `[accepted]` option is set with author `Valentin Noël` de-anonymized (`\icmlauthor{Anonymous}` in the title block but `\icmlcorrespondingauthor{Valentin No\"{e}l}{val.noel@proton.me}` and `\icmlaffiliation{devoteam}{Devoteam, Paris}` immediately below it) — **contradicts anonymity if this is meant to be an anonymous submission draft**, and contradicts "accepted" mode expectations (accepted papers normally aren't anonymous, but the `\icmlauthorlist{Anonymous}` line is dead/stale from a blind-review copy-paste). This is an internal contradiction: `[accepted]` + literal string "Anonymous" as author + real name/email/affiliation below.

## 2. Missing build dependencies — main.tex will not compile
- No `icml2026.sty`, `.cls`, or `.bst` file anywhere in the tree (`find *.sty/*.cls/*.bst` → none). The manuscript is not self-contained/reproducible as tracked.

## 3. No tracked artifact backs any numeric claim
- `.gitignore` explicitly excludes `results/`, `figures/`, `*.pkl`, `*.png` (only `figures/.gitkeep` kept).
- `find results/**` and grep for `.bib` returned nothing tracked.
- Every quantitative number in README.md ("Results" section), docs/laws.md, docs/theory.md, and main.tex (AUC 0.85, p=0.002, Cohen's d_z 1.65/8.2, crystallization C4/C10 table, transfer AUC 0.615/0.804, selectivity S=0.10–0.17, Davis-Kahan θ, DC/AC crossover table) is **asserted in prose/markdown/LaTeX tables with zero corresponding CSV/pickle/log checked into the repo**. There is no `results/<model>/diagnostics_long.csv`, no `results/probe_*.csv`, no `results/pod_scaling/*` committed. A reviewer or auditor cannot verify a single number from the repo state alone — every headline figure is reproduction-dependent on an un-tracked run.
- Recommendation: either commit minimal result CSVs/summary tables backing each cited number, or explicitly label every number in README/docs/main.tex as "illustrative example run, not reproduced in-repo" with a pointer to a script + seed.

## 4. Unsupported "law"/theorem language
- **docs/laws.md**: "Conjecture (Eval-direction crystallization)" is appropriately hedged as a conjecture, but README.md's prose ("Headline: a model's internal representation... is decodably different") and main.tex abstract state the crystallization as if empirically established ("the evaluation-frame shift **crystallizes**...a structure absent below ~1B and monotone within the Qwen...and Llama...families") — while the underlying evidence is **N=15 tasks, single models per rung, no CIs at each scale** (explicitly flagged in docs/theory.md: "The claim currently rests on a single model"). The abstract's confident verb "crystallizes" and the section title `\section{The law: eval-direction crystallization}` overstate a conjecture validated only in-repo as "confirmed clearly in one model and not yet replicated" (docs/theory.md, Validation status table: Qwen-0.5B, Llama-1B, Gemma-2B all fail or are weak; only Llama-3.2-3B clears significance p=0.007).
- Theorems 1 and 2 (DC-invariance, complementary filters) are legitimate proven math (linear algebra on graph Laplacians) — those are fine as stated. But the **P4 falsification** is reported honestly in docs/theory.md yet main.tex §"Theory" folds the falsification into the narrative smoothly without flagging in the abstract that the "sharp prediction" the theory implies was **wrong** — a reader skimming only the abstract could think the theory predicts and confirms the mechanism, when doc/theory.md shows the specific mechanistic account is refuted (p=0.99, 2/15 support).
- **docs/laws.md** "C4 ∈ [0.45,0.65] is ~25–35σ above null" — a σ claim needs the closed-form null variance derivation shown; it's stated but not derived in-repo (only asserted `E[cos]≈0 ± 1/√d`). Flag as an unsupported-without-derivation statistical claim if σ is meant to be taken as rigorous rather than heuristic.

## 5. High-risk numerical claims to gate before any external claim
- README "AUC = 0.85 (95% CI [0.75, 0.98]), p = 0.002" on **N=15 paired tasks, 30 prompts** — small-N; CI width (0.75–0.98) is wide and any redo could shift materially.
- "cross-architecture transfer AUC 0.615, p=0.001" — magnitude is near-chance (paper itself concedes "a poor detector... near-chance in absolute terms"); risk that external readers over-read the p-value as meaning the effect is large.
- Selectivity S = 0.10–0.17 (single metric, N=15) is the load-bearing quantity for the entire "spectral > activation" claim chain; it's from one model (Llama-3.2-3B) and explicitly not yet replicated in the 0.5B/1B/2B rungs (theory.md).
- "Gemma is not a systematic counterexample" is listed as an untested/open falsifiable target (T3) in docs/theory.md, yet main.tex's crystallization table (Table `tab:law`) presents Gemma-2B numbers as part of the confirmed monotone trend — this is an unresolved discrepancy between docs (open question) and main.tex (presented as supporting evidence).

## 6. Stale host/vendor instructions
- `.claude/skills/runpod-connect/SKILL.md` is a live, detailed operational runbook (SSH keys, RunPod Direct-TCP ports, PEP 668 install flags, HF_HOME volume routing, `setsid`/`disown` detach pattern, network-volume quota errors). This is vendor/host-specific infra instruction bundled in-repo; it is fine as an ops runbook but is **not neutral documentation** — it hardcodes a specific commercial GPU rental vendor (RunPod) workflow with example host/port literals (`195.26.233.38:58513`) that read as stale/session-specific artifacts left in a checked-in file. Recommend moving to a README-linked ops doc or marking example values as illustrative placeholders (they already use `<PORT>`/`<HOST>` in most places but the "Quick reference" and step 1 sections mix real example IP/port literals with placeholder syntax inconsistently).
- No stale generic "vendor name" leakage found in README.md or docs/*.md themselves beyond the RunPod skill file.

## 7. Manuscript structure/citations
- All 8 citations in `main.tex`'s `\begin{thebibliography}` are self-referential placeholders: `sad`, `scaling`, `format`, `behaviour`, `notone` are cited as `Anonymous (2026a/b/c/d)` with **no arXiv IDs, DOIs, or venue info** — these are speculative/anonymized future citations, not verifiable references. `scheming` (Meinke et al. 2024) is a real citable work but lacks a venue/arXiv id. `spectraltrust`/`geomreason` are self-cited works by the same author (No\"el) with no venue, i.e. self-citing unpublished/uncitable placeholder works as if they were established literature — a citation-integrity issue if submitted as-is.
- No `.bib` file exists; the bibliography is manually inlined, inconsistent with typical ICML/ICLR bibtex workflows and blocking automated citation checks.

## Proposed machine-readable claim-ledger schema
```json
{
  "claim_id": "string (stable slug, e.g. headline-auc-llama3b)",
  "source_file": "path:line-or-section",
  "claim_text": "verbatim or near-verbatim quote",
  "claim_type": "quantitative | causal | theoretical | conjecture | replication | null-result",
  "quantities": [{"name": "AUC", "value": 0.85, "ci": [0.75, 0.98], "p_value": 0.002, "n": 15}],
  "backing_artifact": "artifact path or ref, or null if untracked",
  "artifact_tracked_in_git": false,
  "reproducibility_status": "reproduced | not-reproduced | single-run-only | conjectural",
  "hedge_level": "proven | strong-evidence | single-model | directional-only | falsified",
  "confound_controls_applied": ["position-match", "length-match", "negative-control"],
  "risk_severity": "low | medium | high",
  "cross_doc_consistency": "consistent | contradicted-by:<file>",
  "reviewer_note": "free text"
}
```

## Proposed pre-results ICLR outline (matching accepted-plan framing, gated on artifacts)
1. Abstract — hedge crystallization as a conjecture pending scale-up, not an established law.
2. Introduction — keep 5 contributions but flag which are theory (proven), which are single-model empirical, which are conjecture.
3. Related Work — cite real anonymized-for-review or genuinely published works only; drop self-citations to unpublished companion papers unless they're concurrently submitted and disclosed as such.
4. Setup — protocol, frames, position-matching (solid, keep as-is).
5. Theory — Theorems 1–2 (keep, proven) + explicit "falsified prediction" subsection (already present in text, make it a first-class subsection, not folded in).
6. Confound-hardened probing results — Table `tab:crossover`, single-model only; add explicit N and CI columns.
7. Crystallization — relabel section title from "The law" to "A candidate scaling regularity (single-family confirmation, conjecture)"; add the Gemma-counterexample caveat inline in the table caption.
8. Cross-architecture transfer — keep, already appropriately hedged with "Calibration" paragraph.
9. Negative/null replications — keep (well-hedged already).
10. Limitations/Falsifiable predictions — keep, this section is a model of honest hedging; other sections should match its tone.
11. Reproducibility — must be upgraded from a promise ("Code, frames, and analysis scripts... are released") to a concrete artifact commitment: pin exact result-file commit/DOI or drop the paragraph.