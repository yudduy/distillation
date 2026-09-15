"""Trusted Stage 1 adapter. Candidate bytes are data, never executable code."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import stat
import sys

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor/sair'
sys.path.insert(0, str(VENDOR))
import httpx
from judge import judge_response
from llm import call_llm
from models import load_models, resolve
from prompt import render_prompt

CONTRACT = 'distill-v1'
MAX_BYTES = 10240
MODEL_IDS = ('gpt-oss-120b', 'llama-3-3-70b-instruct', 'gemma-4-31b-it')
MAX_ATTEMPTS_PER_PAIR = 24
MAX_RESPONSE_BYTES = 4_000_000


@dataclass(frozen=True)
class Route:
    model: str
    endpoint: str
    provider: str
    quantization: str
    reasoning: str | None
    input_usd_per_token: Decimal
    output_usd_per_token: Decimal
    max_input_tokens: int = 131_072
    max_output_tokens: int = 8_192

    @property
    def attempt_ceiling(self) -> Decimal:
        # Reserve the full advertised context as input plus the complete output
        # allowance. This deliberately over-reserves rather than treating a
        # bytes-to-tokens estimate as a monetary bound.
        raw = self.input_usd_per_token * self.max_input_tokens
        raw += self.output_usd_per_token * self.max_output_tokens
        return raw * Decimal('1.10')


ROUTES = {
    'gpt-oss-120b': Route('openai/gpt-oss-120b', 'deepinfra/bf16', 'deepinfra', 'bf16', 'low',
                           Decimal('0.000000037'), Decimal('0.00000017')),
    'llama-3-3-70b-instruct': Route('meta-llama/llama-3.3-70b-instruct', 'deepinfra/turbo', 'deepinfra', 'fp8', None,
                                           Decimal('0.00000010'), Decimal('0.00000032')),
    'gemma-4-31b-it': Route('google/gemma-4-31b-it', 'novita/bf16', 'novita', 'bf16', 'none',
                                    Decimal('0.00000014'), Decimal('0.00000040')),
}

EXECUTION_CONFIG = {
    alias: {
        'model': route.model,
        'endpoint': route.endpoint,
        'quantization': route.quantization,
        'reasoning': route.reasoning,
        'maxInputTokensForBudget': route.max_input_tokens,
        'maxOutputTokens': route.max_output_tokens,
        'inputUsdPerToken': str(route.input_usd_per_token),
        'outputUsdPerToken': str(route.output_usd_per_token),
        'attemptMargin': '1.10',
        'maxAttemptsPerPair': MAX_ATTEMPTS_PER_PAIR,
    }
    for alias, route in ROUTES.items()
}
# Do not let upstream exception bodies, prompts, or responses reach Actions logs.
logging.disable(logging.CRITICAL)


class InvalidRun(Exception):
    """Only fixed, public error codes may leave this process."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_vendor() -> None:
    pins = json.loads((VENDOR / 'SHA256SUMS.json').read_text())
    for name, expected in pins.items():
        if digest((VENDOR / name).read_bytes()) != expected:
            raise InvalidRun('evaluator_pin_mismatch')


def read_prompt(path: Path) -> tuple[str, bytes]:
    # Reject symlinked parents as well as a symlinked final file.
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise InvalidRun('prompt_symlink')
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                raise InvalidRun('prompt_not_regular')
            data = f.read(MAX_BYTES + 1)
        text = data.decode('utf-8', errors='strict')
    except (OSError, UnicodeError):
        raise InvalidRun('prompt_unreadable') from None
    if not 0 < len(data) <= MAX_BYTES:
        raise InvalidRun('prompt_size')
    if any(not any(token in text for token in (f'{{{{equation{i}}}}}', f'{{{{ equation{i} }}}}')) for i in (1, 2)):
        raise InvalidRun('prompt_placeholders')
    return text, data


def load_dataset(path: Path, *, ranked: bool, manifest_path: Path = ROOT / 'ranked-dataset.json'):
    data = path.read_bytes()
    if len(data) > 2_000_000:
        raise InvalidRun('dataset_size')
    version = 'public-smoke-v1'
    if ranked:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('contractVersion') != CONTRACT or not re.fullmatch(r'[a-zA-Z0-9._-]+', manifest.get('version', '')) or digest(data) != manifest.get('sha256'):
            raise InvalidRun('dataset_pin_mismatch')
        version = manifest['version']
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not rows or (ranked and len(rows) != 200):
        raise InvalidRun('dataset_count')
    ids, pairs = set(), set()
    for row in rows:
        if not isinstance(row, dict) or type(row.get('answer')) is not bool:
            raise InvalidRun('dataset_label')
        pair = (row.get('eq1_id'), row.get('eq2_id'))
        if ranked and (any(type(n) is not int or not 1 <= n <= 4694 for n in pair) or pair[0] == pair[1] or pair in pairs):
            raise InvalidRun('dataset_pair')
        pairs.add(pair)
        if not isinstance(row.get('id'), str) or row['id'] in ids:
            raise InvalidRun('dataset_id')
        ids.add(row['id'])
        for key in ('equation1', 'equation2'):
            if not isinstance(row.get(key), str) or not row[key] or len(row[key].encode()) > 8192:
                raise InvalidRun('dataset_equation')
        if ranked and not row.get('provenance'):
            raise InvalidRun('dataset_provenance')
    if ranked and sum(row['answer'] for row in rows) != 100:
        raise InvalidRun('dataset_balance')
    return rows, digest(data), version


def live_settings(env):
    if env.get('DISTILL_LIVE') != '1':
        raise InvalidRun('live_calls_disabled')
    key = env.get('OPENROUTER_API_KEY', '')
    try:
        cap = Decimal(env.get('DISTILL_MAX_SPEND_USD', ''))
    except InvalidOperation:
        raise InvalidRun('spending_cap_required') from None
    if not key or not cap.is_finite() or cap <= 0:
        raise InvalidRun('credentials_or_spending_cap_missing')
    return key, cap


def check_key_budget(data, cap):
    """Require a provider-enforced, non-resetting lifetime ceiling <= approved cap.

    A new dedicated capped key is required for each funded campaign. This also
    bounds retries and in-flight requests independently of local accounting.
    """
    for field in ('limit', 'limit_remaining'):
        value = data.get(field)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise InvalidRun('provider_key_cap_required')
        value = Decimal(str(value))
        if not 0 < value <= cap:
            raise InvalidRun('provider_key_cap_required')
    if data.get('limit_reset') is not None or data.get('include_byok_in_limit') is not True:
        raise InvalidRun('provider_key_cap_required')


class BudgetLedger:
    """Fail-closed local admission and aggregate accounting for HTTP attempts."""

    def __init__(self, cap: Decimal):
        self.cap = cap
        self.confirmed = Decimal(0)
        self.unresolved = Decimal(0)
        self.attempts: dict[str, int] = {}
        self._next_reservation = 0
        self._reservations: dict[int, Decimal] = {}
        self._lock = asyncio.Lock()

    async def reserve(self, route: Route, prompt: str) -> int:
        pair = digest((route.model + '\0' + prompt).encode())
        async with self._lock:
            count = self.attempts.get(pair, 0)
            if count >= MAX_ATTEMPTS_PER_PAIR:
                raise InvalidRun('attempt_limit_exceeded')
            ceiling = route.attempt_ceiling
            if self.confirmed + self.unresolved + ceiling > self.cap:
                raise InvalidRun('local_spending_cap_exceeded')
            self.attempts[pair] = count + 1
            self._next_reservation += 1
            token = self._next_reservation
            self._reservations[token] = ceiling
            self.unresolved += ceiling
            return token

    async def reconcile(self, token: int, cost) -> None:
        if type(cost) not in (int, float, str):
            return
        try:
            amount = Decimal(str(cost))
        except InvalidOperation:
            return
        if not amount.is_finite() or amount < 0:
            return
        async with self._lock:
            reserved = self._reservations.pop(token, None)
            if reserved is None:
                return
            if amount > reserved:
                raise InvalidRun('provider_cost_exceeded_reservation')
            self.unresolved -= reserved
            self.confirmed += amount
            if self.confirmed + self.unresolved > self.cap:
                raise InvalidRun('local_spending_cap_exceeded')

    def metrics(self) -> dict:
        def amount(value: Decimal) -> str:
            return '0' if value == 0 else format(value, 'f')
        return {
            'attempts': sum(self.attempts.values()),
            'confirmedSpendUsd': amount(self.confirmed),
            'unresolvedReservedUsd': amount(self.unresolved),
        }


def base_transport() -> httpx.AsyncBaseTransport:
    return httpx.AsyncHTTPTransport(retries=0)


class GuardedTransport(httpx.AsyncBaseTransport):
    """Normalize and validate the final wire body, then account every POST."""

    def __init__(self, inner: httpx.AsyncBaseTransport, ledger: BudgetLedger):
        self.inner = inner
        self.ledger = ledger

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != 'POST':
            return await self.inner.handle_async_request(request)
        if request.url.scheme != 'https' or request.url.host != 'openrouter.ai' or request.url.path != '/api/v1/chat/completions':
            raise InvalidRun('unexpected_network_target')
        try:
            body = json.loads(request.content)
            route = next(route for route in ROUTES.values() if route.model == body.get('model'))
            messages = body['messages']
            provider = body['provider']
            if (len(messages) != 1 or messages[0].get('role') != 'user' or not isinstance(messages[0].get('content'), str)
                    or body.get('max_tokens') != route.max_output_tokens or body.get('temperature') != 0
                    or body.get('seed') != 0 or provider.get('order') != [route.endpoint]
                    or provider.get('quantizations') != [route.quantization]
                    or provider.get('allow_fallbacks') is not False):
                raise InvalidRun('request_config_mismatch')
            if route.reasoning is None:
                body.pop('reasoning', None)
            elif body.get('reasoning') != {'effort': route.reasoning}:
                raise InvalidRun('request_config_mismatch')
            provider.update({'require_parameters': True, 'data_collection': 'deny', 'zdr': True})
            if set(body) != {'model', 'messages', 'max_tokens', 'temperature', 'seed', 'provider'} | ({'reasoning'} if route.reasoning else set()):
                raise InvalidRun('request_config_mismatch')
        except (KeyError, TypeError, StopIteration, json.JSONDecodeError):
            raise InvalidRun('request_config_mismatch') from None

        reservation = await self.ledger.reserve(route, messages[0]['content'])
        content = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
        headers = request.headers.copy()
        headers['Content-Length'] = str(len(content))
        headers['X-OpenRouter-Cache'] = 'false'
        guarded = httpx.Request(request.method, request.url, headers=headers, content=content, extensions=request.extensions)
        response = await self.inner.handle_async_request(guarded)
        response_body = await response.aread()
        if len(response_body) > MAX_RESPONSE_BYTES:
            raise InvalidRun('provider_response_too_large')
        cost = None
        try:
            usage = json.loads(response_body).get('usage') or {}
            for field, maximum in (('prompt_tokens', route.max_input_tokens), ('completion_tokens', route.max_output_tokens)):
                value = usage.get(field)
                if value is not None and (type(value) is not int or not 0 <= value <= maximum):
                    raise InvalidRun('provider_usage_out_of_bounds')
            cost = usage.get('cost')
        except (AttributeError, json.JSONDecodeError):
            pass
        await self.ledger.reconcile(reservation, cost)
        return httpx.Response(response.status_code, headers=response.headers, content=response_body,
                              extensions=response.extensions, request=guarded)

    async def aclose(self) -> None:
        await self.inner.aclose()


async def evaluate(rows, template, complete, *, timeout=600):
    semaphore = asyncio.Semaphore(6)
    async def one(alias, row):
        async with semaphore:
            response = await asyncio.wait_for(complete(alias, render_prompt(template, row['equation1'], row['equation2'])), timeout)
            if response.finish_reason not in ('stop', 'length', 'content_filter'):
                raise InvalidRun('provider_response_invalid')
            correct, _ = judge_response(response.text if not response.refusal else '', row['answer'])
            return alias, correct, response.tokens_in or 0, response.tokens_out or 0
    tasks = [asyncio.create_task(one(alias, row)) for alias in MODEL_IDS for row in rows]
    try:
        outcomes = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if len(outcomes) != len(rows) * len(MODEL_IDS):
        raise InvalidRun('missing_outcomes')
    models = {}
    for alias in MODEL_IDS:
        results = [outcome for outcome in outcomes if outcome[0] == alias]
        correct = sum(outcome[1] is True for outcome in results)
        models[alias] = {'correct': correct, 'total': len(results), 'accuracy': correct / len(results),
                         'parseFailures': sum(outcome[1] is None for outcome in results)}
    return {'score': sum(m['correct'] for m in models.values()) / len(outcomes),
            'metrics': {'models': models, 'outcomes': len(outcomes), 'questionCount': len(rows),
                        'parseFailures': sum(m['parseFailures'] for m in models.values()),
                        'tokensIn': sum(o[2] for o in outcomes), 'tokensOut': sum(o[3] for o in outcomes)}}


async def live_evaluate(rows, template, env):
    key, cap = live_settings(env)
    configs = load_models(VENDOR / 'evaluation_models.json')
    if set(configs) != set(MODEL_IDS):
        raise InvalidRun('model_config_mismatch')
    ledger = BudgetLedger(cap)
    transport = GuardedTransport(base_transport(), ledger)
    async with httpx.AsyncClient(transport=transport, timeout=180, trust_env=False, follow_redirects=False) as client:
        response = await client.get('https://openrouter.ai/api/v1/key', headers={'Authorization': f'Bearer {key}'})
        response.raise_for_status()
        check_key_budget(response.json()['data'], cap)
        async def complete(alias, prompt_text):
            entry = configs[alias]
            name, provider, kwargs = resolve(entry, api_keys=[key])
            provider.preferred_providers = [ROUTES[alias].endpoint]
            response = await call_llm(client, provider_name=name, provider_config=provider,
                                      model_id=entry.model_id, prompt=prompt_text, kwargs=kwargs)
            expected = 'novita' if alias == 'gemma-4-31b-it' else 'deepinfra'
            if (response.actual_provider or '').lower().split('/')[0] != expected:
                raise InvalidRun('provider_route_mismatch')
            return response
        result = await evaluate(rows, template, complete)
        result['metrics']['billing'] = ledger.metrics()
        return result


async def run(args, env=os.environ, evaluator=live_evaluate):
    score_path = ROOT / 'score.json'
    score_path.unlink(missing_ok=True)
    verify_vendor()
    template, data = read_prompt(ROOT / 'submission/prompt.txt')
    ranked = args.ranked
    if ranked:
        dataset = Path(env.get('DISTILL_PRIVATE_DATASET', ''))
        if not dataset.is_absolute() or dataset.resolve().is_relative_to(ROOT):
            raise InvalidRun('private_dataset_must_be_external')
    else:
        dataset = VENDOR / 'examples/problems_hard3_20.jsonl'
    rows, dataset_hash, version = load_dataset(dataset, ranked=ranked)
    live_settings(env)  # Offline mode is tests, never a fake publishable score.
    candidate = env.get('GITHUB_SHA', '')
    if ranked and not re.fullmatch('[0-9a-f]{40}', candidate):
        raise InvalidRun('candidate_identity_missing')
    result = await evaluator(rows, template, env)
    result['metrics'].update({'contractVersion': CONTRACT, 'datasetKind': 'ranked' if ranked else 'public',
                              'datasetVersion': version, 'datasetSha256': dataset_hash,
                              'configSha256': digest((VENDOR / 'evaluation_models.json').read_bytes()),
                              'executionConfigSha256': digest(json.dumps(EXECUTION_CONFIG, sort_keys=True, separators=(',', ':')).encode()),
                              'promptBytes': len(data), 'promptSha256': digest(data),
                              'candidateSha': candidate, 'runId': env.get('GITHUB_RUN_ID', 'local'),
                              'runAttempt': env.get('GITHUB_RUN_ATTEMPT', 'local')})
    # Atomic publication; only this allowlisted aggregate object is exported.
    temporary = score_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, allow_nan=False, sort_keys=True) + '\n')
    temporary.replace(score_path)
    print(json.dumps({'score': result['score'], 'datasetKind': result['metrics']['datasetKind']}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranked', action='store_true')
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        # Never echo raw provider errors, file paths, dataset fragments or credentials.
        code = str(exc) if isinstance(exc, InvalidRun) else 'evaluation_failed'
        print(f'Math Distillation: {code}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
