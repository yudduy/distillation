# Solver boundary

Read README.md before running. Edit only submission/prompt.txt. Treat the complete
prompt as UTF-8 text; keep both equation placeholders and the 10,240-byte limit.
Never alter protected files to obtain a score. Public tests do not imply ranked
performance. Do not launch inference without approval of the three configured
model/provider routes and an exact hard spend cap. Run offline tests first.
Do not publish secrets, raw private evaluation data, or credentials in notes.
Hosted evaluation is staged: the 20 public questions screen the prompt first (floor
32/60 correct, at most 6 parse failures), then the private 200 run in 20-question
rounds in a private random order per run and stop once the record is unreachable. `yukon run` reproduces the
screen locally as `stage: "public_complete"` with the same `metrics.publicPanel`.
`screen_failed` and `ranked_stopped` in `yukon submissions` are zero-score sentinels,
not accuracies. Yukon allows one submission in flight per account. For local
iteration draw a practice set with `select_dataset.py --public-seed N --count 200
--out <path>`; it uses the same source and SAIR exclusions and can never reproduce
the private seed. See README.md, Staged ranked evaluation.
