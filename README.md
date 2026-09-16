# Math Distillation

An independent Yukon adaptation of SAIR's equational-theories Stage 1 challenge.
Improve a compact complete prompt that helps three fixed language models decide
whether one equation implies another over **all magmas**. A magma has a binary
operation; associativity is not assumed.

## Solver loop

Use Python 3.12. Run `yukon setup` (or `./setup.sh`), then
`.venv/bin/python -m pytest tests vendor/sair/tests` for offline checks.
Edit only `submission/prompt.txt`, retaining `{{equation1}}` / `{{equation2}}`
(or their spaced forms). The entire file must be valid UTF-8 and at most 10,240
bytes. Both placeholders are required. No Python, templates, tools, retrieval,
or arbitrary code are executed from the submission.

Public model testing uses `yukon run` or `./benchmark.sh`. It is **paid** and
requires `DISTILL_LIVE=1`, `OPENROUTER_API_KEY`, and a finite positive
`DISTILL_MAX_SPEND_USD`. Use a dedicated OpenRouter key with a **non-resetting
lifetime credit limit at or below that cap**, including BYOK usage in the limit.
Unlimited, resetting, or larger-cap keys are rejected before inference.
The protected operator workflow may explicitly set `DISTILL_ALLOW_UNCAPPED=1`
to use an unlimited key for an operator-authorized evaluation. The evaluator
accepts this only on `yudduy/distillation` through its benchmark workflow;
local runs cannot select the exception.
Do not paste credentials into notes, prompts, shell arguments, or agent chats.
An agent must obtain approval for the exact model/provider routes and spending authorization
before making paid calls. Setup and tests do not invoke models.

Run `yukon submit`, then inspect `yukon submissions`. Public scores use 20 public
examples and **are not ranked scores**. A local `yukon run` writes
`stage: "public_complete"` with the same `metrics.publicPanel` block the hosted
screen produces (see [Staged ranked evaluation](#staged-ranked-evaluation)), so you
can check the floor before submitting. Ranked evaluation runs in the operator's
workflow. Better verified prompts are automatically promoted to the public shared
baseline; equal scores do not replace it. Your notes can describe your method and
the model you used to author the prompt; those author credits are separate from
the three models being evaluated.

### Iterate locally on a dev set

The public 20 are a smoke test, not a dev set. The private panel is a seeded sample
from Equational Theories at a pinned commit minus every published SAIR problem, and
you can draw practice sets from the same distribution with an explicit public seed:

```sh
.venv/bin/python select_dataset.py --public-seed 42 --count 200 --out /absolute/path/dev-42.jsonl
```

Any non-negative integer seed works; each gives a different balanced set with the same
SAIR exclusions applied. The private selection uses 32 random bytes, so no public seed
reproduces it. The file loads through the evaluator's public validator
(`load_dataset(path, ranked=False)`) and can be scored with your own key through
`evaluate.evaluate()` or `live_evaluate()` from Python. No model calls occur while
selecting.


### Submit your insight

After editing `submission/prompt.txt`, write `submission-note.md` (at least 5 KiB)
explaining your method, what changed, the tests you ran, and limitations. Notes are
public; keep credentials and private data out. Replace the model and harness
placeholders with the exact authoring model and coding tool you actually used:

```sh
yukon submit --model "YOUR_EXACT_MODEL" --harness "YOUR_CODING_TOOL" --note-file submission-note.md
yukon submissions
```

The service runs the ranked evaluation. You do not need an OpenRouter key to submit;
you need your own key only for optional local model testing. A queued submission
appears under your Yukon account. Once a strict improvement is validated and
promoted, the shared best and leaderboard update. Use `yukon sync` from a clean
checkout to start from the latest promoted insight; preserve your work first.

## Frozen v2 contract

| Model | OpenRouter route | Reasoning |
| --- | --- | --- |
| `openai/gpt-oss-120b` | DeepInfra turbo, bf16 | low |
| `meta-llama/llama-3.3-70b-instruct` | DeepInfra turbo, fp8 | disabled |
| `google/gemma-4-31b-it` | Novita, bf16 | disabled |

All use temperature 0, seed 0, 8,192 output tokens, no fallback providers, and
one complete user prompt per question. Preserve the pinned upstream retry behavior,
including one retry of an empty non-refusal response; every attempt remains under
the token cap. Each model/question has a ten-minute provider-call deadline after
queue admission. There are six requests globally, capped at 1 GPT, 3 Llama, and
6 Gemma calls at once, one ranked workflow at a time, and a six-hour job ceiling.
A seed and zero temperature do not guarantee bit-for-bit provider reproducibility.

The trusted adapter sends exact endpoint tag `deepinfra/turbo` for GPT and
Llama, and `novita/bf16` for Gemma; it omits the unsupported reasoning field
only for Llama. It also sends route-specific maximum input/output prices.
Before each HTTP attempt it reserves a conservative uncached
input/output ceiling against the finite local run cap when configured and permits at most 24 attempts
per model/prompt. Reported costs reconcile reservations; missing or invalid cost
data remains fully reserved and can stop further admission.

Set `DISTILL_PRIVATE_RECEIPTS` to a new absolute path outside the checkout to
write one sanitized JSONL record per completion HTTP attempt. The evaluator
creates it with mode 0600 and refuses existing files. Receipts contain opaque
attempt/reservation IDs, route, status, elapsed time, token/cost coverage, and
BYOK status only. They are private ephemeral accounting data and must never be
uploaded with `score.json`; the ranked workflow removes them during cleanup.

Hosted runs emit a progress event after every 20 completed responses per model.
They also upload `evaluation-costs.json` on success or failure with sanitized
per-model cost, token, timing, and HTTP-status aggregates. It contains no questions,
answers, responses, credentials, or receipt identifiers and is separate from the
success-only `score.json` artifact.

The ranked panel has 100 TRUE and 100 FALSE questions. All three models answer all
200. `score = correct / 600`, in [0, 1]. Wrong answers, refusals, and unparseable
completed answers count as incorrect. Valid parseable answers at the token limit
are graded normally. Provider errors, timeouts, missing outcomes, unexpected routes,
and dataset/configuration mismatches fail the run with no score. A stale score is
removed before evaluation. Only the trusted adapter writes the final aggregate.

## Staged ranked evaluation

One hosted run has two stages inside the same workflow, ledger, and client. Yukon
sees a single `score.json`; `metrics.stage` says which stage ended the run.

1. **Screen.** The 20 public questions × 3 models (60 calls, about three minutes) run
   first. The floor is `DISTILL_SCREEN_MIN_CORRECT` = **32/60** correct and at most
   `DISTILL_SCREEN_MAX_PARSE_FAILURES` = **6** parse failures. A prompt below the floor
   writes `stage: "screen_failed"`, `score: 0`, `partial: true`, and spends nothing
   on the private panel. The floor is deliberately loose: it removes broken prompts;
   it does not order good ones (the public panel mis-ordered the current record).
2. **Ranked rounds.** The 200 private questions are put in a private random order
   drawn fresh for each run (never recorded anywhere) and evaluated in
   `DISTILL_ROUND_QUESTIONS` = **20**-question rounds of 60 outcomes. At each round boundary, with nothing in flight, the run
   compares `correct so far + outcomes remaining` against the record fetched from the
   public benchmark row at run start (`DISTILL_RECORD_SOURCE_URL`). When even a perfect
   finish could not exceed the record, no further round is admitted and the run writes
   `stage: "ranked_stopped"`, `score: 0`, `partial: true`, and
   `metrics.stopped = {roundsCleared, roundsTotal, roundQuestions, completedOutcomes,
   correctSoFar, recordCorrect}`. A survivor of all ten rounds has a complete
   600-outcome score and `stage: "ranked_complete"`.

`score` is non-zero only for `ranked_complete`. Yukon treats the zero-score sentinels
as ordinary non-improving results (**rejected**, never *failed*), so `yukon submissions`
shows them alongside the stage, the public-panel counts, and the rounds cleared.
Feedback is Ladder-style on purpose: a non-improving submission learns its screen
count, how many rounds it cleared, and that it did not beat the record; only promoted
runs publish a full ranked score. Yukon also admits **one submission in flight per
account**; a second `yukon submit` while one is validating is refused with HTTP 409.

Every stage carries `metrics.publicPanel`: the 20 public problems copied verbatim
from `vendor/sair/examples/problems_hard3_20.jsonl` (`{id, eq1Id, eq2Id, equation1,
equation2, expected}`), one 20-character verdict string per model (`1` correct, `0`
wrong, `u` unparseable, index `i` is `problems[i]`), per-model counts, the screen's
own token totals, and the floor result. The top-level `outcomes`, `parseFailures`,
`tokensIn`, and `tokensOut` count the ranked stage only. A local `yukon run` emits the identical block with `stage: "public_complete"`.
Per-question data exists **only** for the public panel; ranked rows never reach a
per-question field, and a sentinel `score.json` is never an accuracy.

`metrics.evaluationPolicy` records the floor, the round size, the order policy
(`shuffleSeed: "private-random"`), whether stopping was on, and the record used (`{source, score, correct}`, or `"unavailable"` when the fetch
failed, or `"disabled"`). The record fetch is unauthenticated, requires the
benchmark's `direction` to be `+`, and **fails open**: any error means no stopping
and one full-price run rather than a wave of failed submissions. Set
`DISTILL_STOP_WHEN_IMPOSSIBLE=0` to disable stopping entirely. The screen also runs
before every ranked evaluation and asserts that no public `(eq1_id, eq2_id)` pair
appears in the private panel; `prepare_private.py` makes the same check at fixture
load, before any spend.

None of this changes `distill-v2`: generation settings, routes, the parser, the dataset
pin, and complete-run scoring are unchanged, and `configSha256` /
`executionConfigSha256` are byte-identical across all four stages. The policy lives
only in `metrics.evaluationPolicy` and `metrics.publicPanel`. `benchmark.json` sets
`minScoreImprovementBips: 125` (about six outcomes at the current record) because
600 outcomes over 200 questions carry a sampling error near 1.7 pp; a promotion must
clear the noise floor, not just one outcome.

The model configurations and parser come from SAIR's Apache-2.0 judge at
`fe00cf9e9080dba6634882c9316b73d536c4fe60`. `vendor/sair/` contains unchanged source,
examples, license, and tests; `SHA256SUMS.json` checks their bytes. The verdict parser
prefers boxed answers over labeled answers over bare first/last-line answers;
within one marker type the last occurrence wins. See the upstream README for details.

## Operator bring-up (not performed by this bundle)

1. Copy this directory's contents to the private operator repository
   `yudduy/distillation`. Keep every path except
   `submission/prompt.txt` protected by the manifest. Set branch controls so solvers
   cannot push evaluator/workflow changes. Verify Apps and source identity against
   Yukon's `docs/github-actions-benchmark-author-guide.md` before import.
2. Run `.venv/bin/python select_dataset.py --output-dir /absolute/private/new-directory`.
   The destination must be new and outside both this bundle and its enclosing Git
   repositories. This downloads pinned public data, uses a fresh private 32-byte
   seed, and stores files with private permissions. `--seed-file` reproduces a
   selection in a different new directory. No model calls occur.
3. The selection uses proven all-magma implications and proven counterexamples from
   Equational Theories `1aec8a7acf223b7c56e4830977b6e90d4ef1924b`. It excludes
   conjectures, self-pairs, conflicts, and all published SAIR pairs at
   `cf2e964ae911e21421bc9dbf7e28cc8df7291983`, including published evaluation data.
   Each label retains its source theorem/file/line. Selection is deterministic HMAC
   ordering within each class, with classes interleaved. This is a private selection
   of public mathematics, not evidence of unseen mathematics or SAIR score parity.
4. Copy only the generated **public** `ranked-dataset.json` to this repository.
   Its checked-in empty hash deliberately blocks ranked runs until then. Keep the
   seed, questions, and provenance outside participant access. Gzip/base64 the
   `ranked.jsonl` bytes into the GitHub `ranked` environment's
   `DISTILL_PRIVATE_DATA_GZIP_BASE64` secret. Confirm the compressed payload fits
   GitHub's secret size limit; never put it in repository files or workflow artifacts.
5. Set `ranked` environment secret `OPENROUTER_API_KEY` and variable
   `DISTILL_LIVE=1` only after funding approval. For ordinary capped runs, also set
   `DISTILL_MAX_SPEND_USD`; allow about $0.10 more than a bare ranked run for the
   60 screen calls. All of these, and the staged-evaluation variables below, live in
   the GitHub **environment** named `ranked` (the workflow declares
   `environment: ranked`), not at repository level. Set `DISTILL_RECORD_SOURCE_URL`
   to the public benchmark row, `https://<yukon-api>/api/benchmarks/<benchmark-id>`,
   so stopping can fire; its `direction` must be `+`. `DISTILL_SCREEN_MIN_CORRECT`,
   `DISTILL_SCREEN_MAX_PARSE_FAILURES`, `DISTILL_STOP_WHEN_IMPOSSIBLE`, and
   `DISTILL_ROUND_QUESTIONS` are optional and default to 32, 6, on, and 20.
   Normally the provider key's non-resetting credit limit is the campaign ceiling.
   For an explicitly authorized uncapped hosted evaluation, set protected variable
   `DISTILL_ALLOW_UNCAPPED=1`. The evaluator accepts that exception only from the
   `yudduy/distillation` benchmark workflow. A successful hosted response must
   explicitly report non-BYOK billing; local emulation fails closed.
6. Install the Yukon dev GitHub App. Validate the starter prompt through the ranked
   workflow before importing/opening in Yukon dev. Explicitly leave
   `claimedScoreEnabled=false` because local and official datasets differ. The
   manifest's automatic promotion mode retains strictly improving candidates.
7. Pin the resulting benchmark UUID as `NEXT_PUBLIC_DISTILL_BENCHMARK_REF` in the
   challenges UI and include `distill` in its live-board allowlist only when ready.
   Obtain challenge-specific legal copy before public launch. Exercise submit,
   validation, rejection, and promotion in dev before any production rollout.

Do not publish logs containing private questions, expected labels, responses,
private paths, or raw HTTP error bodies. Only `score.json` and the sanitized
`evaluation-costs.json` accounting artifact are uploaded. The score contains
aggregate per-model accuracy, parse-failure counts, token counts, prompt size/hash,
candidate SHA, run ID, dataset/configuration identities, the stage, the public-panel
block, the evaluation policy, and, for a stopped run, the round counts. The only
per-question data it ever carries is the public 20-question panel. A sentinel
(`screen_failed`, `ranked_stopped`) is a zero score with `partial: true`, never an
accuracy, and never ranked per-question data. `evaluation-costs.json` reports
`status` as `succeeded`, `screen_failed`, `stopped`, or `failed`.
No participant PII or provider credentials belong in this artifact.

`distill-v2` is immutable. It retains the v1 panel, parser, and generation
settings while pinning GPT to the working DeepInfra BF16 turbo endpoint.
Changing the panel, parser, model route, or scoring
configuration requires a new contract/dataset version and reevaluation before
scores can be compared. Published aggregate feedback enables adaptation to this
panel over time; it is a continuous optimization benchmark, not a fresh final test.

## Attribution and licensing

SAIR Stage 1 judge: https://github.com/SAIRcompetition/equational-theories-stage1-judge

Equational Theories: https://github.com/teorth/equational_theories

Public SAIR data: https://huggingface.co/datasets/SAIRfoundation/equational-theories-selected-problems

The upstream judge remains under `vendor/sair/LICENSE`. Operator-authored files
are licensed under Apache-2.0; see `LICENSE`. Preserve these notices when copying.

The starter `submission/prompt.txt` is the exact public SAIR Stage 1 prompt
attributed to **Dufius**, the highest-average entry in the imported 2026-04-30
snapshot (`3,733 / 5,400`, 69.1296%). Its source record is
`apps/challenges-ui/public/distill-data/sair-stage1-2026-04-30/imb_018d8459bd6c4832bf527b25.json`
in the Yukon source checkout. The prompt is 10,068 UTF-8 bytes with SHA-256
`4a92dfdfbe555853cc4763b40aba667209eae421acbd139fdfd42d59a8dab459`.
It is reused with contributor and SAIR attribution; the imported research notes
record that participant-prompt republication terms remain unclear, so resolve
that licensing question before publishing the operator repository.
