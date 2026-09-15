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
examples and **are not ranked scores**. Ranked evaluation runs in the operator's
workflow. Better verified prompts are automatically promoted to the public shared
baseline; equal scores do not replace it. Your notes can describe your method and
the model you used to author the prompt; those author credits are separate from
the three models being evaluated.


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
   `DISTILL_MAX_SPEND_USD`.
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

Do not publish partial results or logs containing questions, expected labels,
responses, private paths, or raw HTTP error bodies. Only `score.json` and the
sanitized `evaluation-costs.json` accounting artifact are uploaded. The score contains
aggregate per-model accuracy, parse-failure counts, token counts,
prompt size/hash, candidate SHA, run ID, and dataset/configuration identities.
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
