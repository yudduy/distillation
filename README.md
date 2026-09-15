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
Do not paste credentials into notes, prompts, shell arguments, or agent chats.
An agent must obtain approval for the exact model/provider routes and spend cap
before making paid calls. Setup and tests do not invoke models.

Run `yukon submit`, then inspect `yukon submissions`. Public scores use 20 public
examples and **are not ranked scores**. Ranked evaluation runs in the operator's
workflow. Better verified prompts are automatically promoted to the public shared
baseline; equal scores do not replace it. Your notes can describe your method and
the model you used to author the prompt; those author credits are separate from
the three models being evaluated.

## Frozen v1 contract

| Model | OpenRouter route | Reasoning |
| --- | --- | --- |
| `openai/gpt-oss-120b` | DeepInfra, bf16 | low |
| `meta-llama/llama-3.3-70b-instruct` | DeepInfra, fp8 | disabled |
| `google/gemma-4-31b-it` | Novita, bf16 | disabled |

All use temperature 0, seed 0, 8,192 output tokens, no fallback providers, and
one complete user prompt per question. Preserve the pinned upstream retry behavior,
including one retry of an empty non-refusal response; every attempt remains under
the token cap. Each model/question has a ten-minute outer deadline. There are six
concurrent requests, one ranked workflow at a time, and a six-hour job ceiling.
A seed and zero temperature do not guarantee bit-for-bit provider reproducibility.

The trusted adapter sends exact endpoint tags `deepinfra/bf16`,
`deepinfra/turbo`, and `novita/bf16`; it omits the unsupported reasoning field
only for Llama. Before each HTTP attempt it reserves a conservative uncached
input/output ceiling against the local run cap and permits at most 24 attempts
per model/prompt. Reported costs reconcile reservations; missing or invalid cost
data remains fully reserved and can stop further admission.

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
5. Set `ranked` environment secret `OPENROUTER_API_KEY` and variables
   `DISTILL_LIVE=1` and `DISTILL_MAX_SPEND_USD` only after funding approval.
   The provider key's non-resetting credit limit is the campaign's enforced ceiling;
   exhaustion stops scoring. Replenishment is a new funding decision, not an
   automatic retry. See https://openrouter.ai/docs/api_reference/limits.
6. Install the Yukon dev GitHub App. Validate the starter prompt through the ranked
   workflow before importing/opening in Yukon dev. Explicitly leave
   `claimedScoreEnabled=false` because local and official datasets differ. The
   manifest's automatic promotion mode retains strictly improving candidates.
7. Pin the resulting benchmark UUID as `NEXT_PUBLIC_DISTILL_BENCHMARK_REF` in the
   challenges UI and include `distill` in its live-board allowlist only when ready.
   Obtain challenge-specific legal copy before public launch. Exercise submit,
   validation, rejection, and promotion in dev before any production rollout.

Do not publish partial results or logs containing questions, expected labels,
responses, private paths, or raw HTTP error bodies. Only `score.json` is uploaded.
It contains aggregate per-model accuracy, parse-failure counts, token counts,
prompt size/hash, candidate SHA, run ID, and dataset/configuration identities.
No participant PII or provider credentials belong in this artifact.

`distill-v1` is immutable. Changing the panel, parser, model route, or scoring
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
