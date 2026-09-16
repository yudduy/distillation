"""Trusted Stage 1 adapter. Candidate bytes are data, never executable code."""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import secrets
import stat
import sys
import time

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor/sair'
sys.path.insert(0, str(VENDOR))
import httpx
from judge import judge_response
from llm import call_llm
from models import load_models, resolve
from prompt import render_prompt

CONTRACT = 'distill-v2'
MAX_BYTES = 10240
MODEL_IDS = ('gpt-oss-120b', 'llama-3-3-70b-instruct', 'gemma-4-31b-it')
MODEL_CONCURRENCY = dict(zip(MODEL_IDS, (1, 3, 6), strict=True))
MAX_ATTEMPTS_PER_PAIR = 24
MAX_RESPONSE_BYTES = 4_000_000
HOSTED_REPOSITORY = 'yudduy/distillation'
# Staged evaluation. The public 20-question panel screens a prompt before any private spend; the
# ranked panel is then evaluated in shuffled rounds and stopped once the record is unreachable.
SCREEN_DATASET = VENDOR / 'examples/problems_hard3_20.jsonl'
SCREEN_DEFAULT_MIN_CORRECT = 32
SCREEN_DEFAULT_MAX_PARSE_FAILURES = 6
ROUND_DEFAULT_QUESTIONS = 20
STAGES = ('public_complete', 'screen_failed', 'ranked_stopped', 'ranked_complete')
RECORD_SOURCE = 'yukon-public-api'


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
    'gpt-oss-120b': Route('openai/gpt-oss-120b', 'deepinfra/turbo', 'deepinfra', 'bf16', 'low',
                           Decimal('0.00000015'), Decimal('0.00000060')),
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


@dataclass(frozen=True)
class Reservation:
    attempt_id: str
    reservation_id: str
    alias: str
    amount: Decimal


class PrivateReceipts:
    """Append sanitized attempt receipts to one newly created private file."""

    def __init__(self, path: Path):
        if (not path.is_absolute() or path.resolve().is_relative_to(ROOT.resolve())
                or any(parent.is_symlink() for parent in (path, *path.parents))):
            raise InvalidRun('private_receipts_must_be_external')
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.fchmod(fd, 0o600)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise InvalidRun('private_receipts_unavailable')
        except OSError:
            raise InvalidRun('private_receipts_unavailable') from None
        self._file = os.fdopen(fd, 'wb', buffering=0)
        self._lock = asyncio.Lock()

    async def write(self, receipt: dict) -> None:
        data = (json.dumps(receipt, allow_nan=False, sort_keys=True, separators=(',', ':')) + '\n').encode()
        async with self._lock:
            self._file.write(data)
            os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


def private_receipts(env) -> PrivateReceipts | None:
    value = env.get('DISTILL_PRIVATE_RECEIPTS', '').strip()
    return PrivateReceipts(Path(value)) if value else None


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
    if uncapped_hosted_mode(env):
        if not key:
            raise InvalidRun('credentials_or_spending_cap_missing')
        return key, Decimal('Infinity')
    try:
        cap = Decimal(env.get('DISTILL_MAX_SPEND_USD', ''))
    except InvalidOperation:
        raise InvalidRun('spending_cap_required') from None
    if not key or not cap.is_finite() or cap <= 0:
        raise InvalidRun('credentials_or_spending_cap_missing')
    return key, cap


def hosted_workflow(env) -> bool:
    workflow_prefix = f'{HOSTED_REPOSITORY}/.github/workflows/benchmark.yml@'
    return (env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == HOSTED_REPOSITORY
            and env.get('GITHUB_WORKFLOW_REF', '').startswith(workflow_prefix))


def uncapped_hosted_mode(env) -> bool:
    """Allow an unlimited key only in the protected operator workflow."""
    if env.get('DISTILL_ALLOW_UNCAPPED') != '1':
        return False
    if not hosted_workflow(env):
        raise InvalidRun('uncapped_hosted_context_invalid')
    return True


def _policy_int(env, name, default, low, high, code):
    raw = env.get(name, '')
    raw = raw.strip() if isinstance(raw, str) else raw
    if raw == '' or raw is None:
        return default
    if not isinstance(raw, str) or not re.fullmatch(r'[0-9]+', raw) or not low <= int(raw) <= high:
        raise InvalidRun(code)
    return int(raw)


def screen_policy(env) -> tuple[int, int]:
    """Public-panel floor: (minimum correct of 60, maximum parse failures of 60)."""
    return (_policy_int(env, 'DISTILL_SCREEN_MIN_CORRECT', SCREEN_DEFAULT_MIN_CORRECT, 0, 60, 'screen_policy_invalid'),
            _policy_int(env, 'DISTILL_SCREEN_MAX_PARSE_FAILURES', SCREEN_DEFAULT_MAX_PARSE_FAILURES, 0, 60, 'screen_policy_invalid'))


def stop_policy(env) -> tuple[bool, str | None, int]:
    """(stop when the record is unreachable, record source URL, questions per round)."""
    switch = env.get('DISTILL_STOP_WHEN_IMPOSSIBLE', '').strip()
    if switch not in ('', '0', '1'):
        raise InvalidRun('stop_policy_invalid')
    url = env.get('DISTILL_RECORD_SOURCE_URL', '').strip() or None
    rounds = _policy_int(env, 'DISTILL_ROUND_QUESTIONS', ROUND_DEFAULT_QUESTIONS, 1, 200, 'round_policy_invalid')
    return switch != '0', url, rounds


def record_correct_from_score(score, total=600) -> int | None:
    """Exact correct count behind a published score, or None when it is not k/total."""
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        return None
    count = round(score * total)
    return count if abs(count / total - score) < 1e-9 else None


def record_transport() -> httpx.AsyncBaseTransport:
    return httpx.AsyncHTTPTransport(retries=0)


async def fetch_record(url) -> dict | None:
    """Read the public benchmark row's current best. Fails open: any problem means no stopping."""
    try:
        if not isinstance(url, str) or not url.startswith('https://'):
            raise ValueError('record_url')
        async with httpx.AsyncClient(transport=record_transport(), timeout=20, trust_env=False,
                                     follow_redirects=False) as client:
            response = await client.get(url)
        if response.status_code != 200:
            raise ValueError('record_status')
        benchmark = response.json()['benchmark']
        if benchmark.get('direction') != '+':  # Yukon's enum is "+" | "-"; anything else fails open.
            raise ValueError('record_direction')
        score = benchmark['currentBestScore']
        correct = record_correct_from_score(score)
        if correct is None:
            raise ValueError('record_score')
        return {'source': RECORD_SOURCE, 'score': score, 'correct': correct}
    except Exception:
        # Never surface the body or the exception; an outage costs one full-price run, not a failed one.
        return None


def private_shuffle_seed() -> str:
    """Fresh per run and never recorded: a reproducible order would let repeated stopped runs
    accumulate linear constraints on per-question private results."""
    return secrets.token_hex(16)


def assert_screen_disjoint(screen_rows, ranked_rows) -> None:
    """The public screen and the private ranked panel must not share an equation pair."""
    pairs = {(row['eq1_id'], row['eq2_id']) for row in screen_rows}
    if any((row.get('eq1_id'), row.get('eq2_id')) in pairs for row in ranked_rows):
        raise InvalidRun('screen_panel_overlap')


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
        self._reservations: dict[str, Reservation] = {}
        self._models = {
            alias: {'attempts': 0, 'completed': 0, 'failed': 0, 'transportFailures': 0,
                    'usageReported': 0, 'costReported': 0, 'tokensIn': 0, 'tokensOut': 0,
                    'reasoningTokens': 0, 'cachedTokens': 0,
                    'tokensInReported': 0, 'tokensOutReported': 0,
                    'reasoningTokensReported': 0, 'cachedTokensReported': 0,
                    'isByokReported': 0, 'byokAttempts': 0, 'confirmed': Decimal(0),
                    'unresolved': Decimal(0), 'elapsedTotal': 0.0, 'elapsedMax': 0.0,
                    'statuses': {}}
            for alias in ROUTES
        }
        self._lock = asyncio.Lock()

    async def reserve(self, route: Route, prompt: str) -> Reservation:
        pair = digest((route.model + '\0' + prompt).encode())
        alias = next(alias for alias, candidate in ROUTES.items() if candidate is route)
        async with self._lock:
            count = self.attempts.get(pair, 0)
            if count >= MAX_ATTEMPTS_PER_PAIR:
                raise InvalidRun('attempt_limit_exceeded')
            ceiling = route.attempt_ceiling
            if self.confirmed + self.unresolved + ceiling > self.cap:
                raise InvalidRun('local_spending_cap_exceeded')
            self.attempts[pair] = count + 1
            reservation = Reservation(secrets.token_hex(16), secrets.token_hex(16), alias, ceiling)
            self._reservations[reservation.reservation_id] = reservation
            self.unresolved += ceiling
            model = self._models[alias]
            model['attempts'] += 1
            model['unresolved'] += ceiling
            return reservation

    async def finish(self, reservation: Reservation, *, status: int | None, elapsed_ms: float,
                     usage: dict | None, outcome: str, receipts: PrivateReceipts | None) -> None:
        cost = None if usage is None else usage.get('cost')
        amount: Decimal | None = None
        if type(cost) in (int, float, str):
            try:
                candidate = Decimal(str(cost))
                if candidate.is_finite() and candidate >= 0:
                    amount = candidate
            except InvalidOperation:
                pass
        violation = None
        async with self._lock:
            current = self._reservations.get(reservation.reservation_id)
            if current is None:
                return
            model = self._models[reservation.alias]
            if amount is not None:
                self._reservations.pop(reservation.reservation_id)
                self.unresolved -= reservation.amount
                model['unresolved'] -= reservation.amount
                self.confirmed += amount
                model['confirmed'] += amount
                model['costReported'] += 1
                if amount > reservation.amount:
                    violation = 'provider_cost_exceeded_reservation'
            if self.confirmed + self.unresolved > self.cap:
                violation = 'local_spending_cap_exceeded'
            model['elapsedTotal'] += elapsed_ms
            model['elapsedMax'] = max(model['elapsedMax'], elapsed_ms)
            model['completed' if outcome == 'completed' else 'failed'] += 1
            if outcome in ('transport_error', 'cancelled'):
                model['transportFailures'] += 1
            if status is not None:
                key = str(status)
                model['statuses'][key] = model['statuses'].get(key, 0) + 1
            if usage is not None:
                model['usageReported'] += 1
                for target, reported, source in (
                    ('tokensIn', 'tokensInReported', 'tokens_in'),
                    ('tokensOut', 'tokensOutReported', 'tokens_out'),
                    ('reasoningTokens', 'reasoningTokensReported', 'reasoning_tokens'),
                    ('cachedTokens', 'cachedTokensReported', 'cached_tokens'),
                ):
                    value = usage.get(source)
                    if value is not None:
                        model[target] += value
                        model[reported] += 1
                if usage.get('is_byok') is not None:
                    model['isByokReported'] += 1
                    model['byokAttempts'] += usage['is_byok'] is True

        if receipts is not None:
            await receipts.write({
                'attemptId': reservation.attempt_id,
                'reservationId': reservation.reservation_id,
                'model': ROUTES[reservation.alias].model,
                'endpoint': ROUTES[reservation.alias].endpoint,
                'httpStatus': status,
                'elapsedMs': round(elapsed_ms, 3),
                'outcome': outcome,
                'tokensIn': None if usage is None else usage.get('tokens_in'),
                'tokensOut': None if usage is None else usage.get('tokens_out'),
                'reasoningTokens': None if usage is None else usage.get('reasoning_tokens'),
                'cachedTokens': None if usage is None else usage.get('cached_tokens'),
                'isByok': None if usage is None else usage.get('is_byok'),
                'actualCostUsd': None if amount is None else format(amount, 'f'),
                'unresolvedReservationUsd': format(reservation.amount, 'f') if amount is None else '0',
            })
        if violation is not None:
            raise InvalidRun(violation)

    def metrics(self) -> dict:
        def amount(value: Decimal) -> str:
            return '0' if value == 0 else format(value, 'f')
        models = {}
        for alias, values in self._models.items():
            elapsed_total = values['elapsedTotal']
            models[alias] = {
                'attempts': values['attempts'],
                'completedAttempts': values['completed'],
                'failedAttempts': values['failed'],
                'transportFailures': values['transportFailures'],
                'usageReportedAttempts': values['usageReported'],
                'costReportedAttempts': values['costReported'],
                'tokensIn': values['tokensIn'],
                'tokensOut': values['tokensOut'],
                'reasoningTokens': values['reasoningTokens'],
                'cachedTokens': values['cachedTokens'],
                'tokensInReportedAttempts': values['tokensInReported'],
                'tokensOutReportedAttempts': values['tokensOutReported'],
                'reasoningTokensReportedAttempts': values['reasoningTokensReported'],
                'cachedTokensReportedAttempts': values['cachedTokensReported'],
                'isByokReportedAttempts': values['isByokReported'],
                'byokAttempts': values['byokAttempts'],
                'confirmedSpendUsd': amount(values['confirmed']),
                'unresolvedReservedUsd': amount(values['unresolved']),
                'elapsedMsTotal': round(elapsed_total, 3),
                'elapsedMsMean': round(elapsed_total / values['attempts'], 3) if values['attempts'] else 0,
                'elapsedMsMax': round(values['elapsedMax'], 3),
                'httpStatusCounts': dict(sorted(values['statuses'].items())),
            }
        return {
            'attempts': sum(self.attempts.values()),
            'confirmedSpendUsd': amount(self.confirmed),
            'unresolvedReservedUsd': amount(self.unresolved),
            'models': models,
        }


def write_cost_artifact(ledger: BudgetLedger, status: str) -> None:
    path = ROOT / 'evaluation-costs.json'
    temporary = path.with_suffix('.json.tmp')
    payload = {'contractVersion': CONTRACT, 'status': status, 'billing': ledger.metrics()}
    temporary.write_text(json.dumps(payload, allow_nan=False, sort_keys=True) + '\n')
    temporary.replace(path)


def base_transport() -> httpx.AsyncBaseTransport:
    return httpx.AsyncHTTPTransport(retries=0)


class GuardedTransport(httpx.AsyncBaseTransport):
    """Normalize and validate the final wire body, then account every POST."""

    def __init__(self, inner: httpx.AsyncBaseTransport, ledger: BudgetLedger,
                 receipts: PrivateReceipts | None = None, reject_byok: bool = False):
        self.inner = inner
        self.ledger = ledger
        self.receipts = receipts
        self.reject_byok = reject_byok

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
            provider.update({
                'require_parameters': True,
                'data_collection': 'deny',
                'zdr': True,
                'max_price': {
                    'prompt': float(route.input_usd_per_token * 1_000_000),
                    'completion': float(route.output_usd_per_token * 1_000_000),
                    'request': 0,
                },
            })
            if set(body) != {'model', 'messages', 'max_tokens', 'temperature', 'seed', 'provider'} | ({'reasoning'} if route.reasoning else set()):
                raise InvalidRun('request_config_mismatch')
        except (KeyError, TypeError, StopIteration, json.JSONDecodeError):
            raise InvalidRun('request_config_mismatch') from None

        reservation = await self.ledger.reserve(route, messages[0]['content'])
        started = time.monotonic()
        content = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
        headers = request.headers.copy()
        headers['Content-Length'] = str(len(content))
        headers['X-OpenRouter-Cache'] = 'false'
        guarded = httpx.Request(request.method, request.url, headers=headers, content=content, extensions=request.extensions)
        try:
            response = await self.inner.handle_async_request(guarded)
            response_body = await response.aread()
        except BaseException as exc:
            outcome = 'cancelled' if isinstance(exc, asyncio.CancelledError) else 'transport_error'
            await self.ledger.finish(reservation, status=None, elapsed_ms=(time.monotonic() - started) * 1000,
                                     usage=None, outcome=outcome, receipts=self.receipts)
            raise

        status = response.status_code
        elapsed_ms = (time.monotonic() - started) * 1000
        if len(response_body) > MAX_RESPONSE_BYTES:
            await self.ledger.finish(reservation, status=status, elapsed_ms=elapsed_ms, usage=None,
                                     outcome='response_too_large', receipts=self.receipts)
            raise InvalidRun('provider_response_too_large')
        usage = None
        try:
            raw_usage = json.loads(response_body).get('usage')
            if raw_usage is not None:
                usage = {
                    'tokens_in': raw_usage.get('prompt_tokens'),
                    'tokens_out': raw_usage.get('completion_tokens'),
                    'reasoning_tokens': (raw_usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                    'cached_tokens': (raw_usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
                    'is_byok': raw_usage.get('is_byok'),
                    'cost': raw_usage.get('cost'),
                }
                bounded_usage = (
                    (usage['tokens_in'], route.max_input_tokens),
                    (usage['tokens_out'], route.max_output_tokens),
                    (usage['reasoning_tokens'], route.max_output_tokens),
                    (usage['cached_tokens'], route.max_input_tokens),
                )
                if (any(value is not None and (type(value) is not int or not 0 <= value <= maximum)
                        for value, maximum in bounded_usage)
                        or (usage['is_byok'] is not None and type(usage['is_byok']) is not bool)):
                    await self.ledger.finish(reservation, status=status, elapsed_ms=elapsed_ms, usage=None,
                                             outcome='usage_out_of_bounds', receipts=self.receipts)
                    raise InvalidRun('provider_usage_out_of_bounds')
        except (AttributeError, json.JSONDecodeError):
            usage = None
        if self.reject_byok and 200 <= status < 300 and (usage is None or usage.get('is_byok') is not False):
            await self.ledger.finish(reservation, status=status, elapsed_ms=elapsed_ms, usage=usage,
                                     outcome='hosted_billing_mode_invalid', receipts=self.receipts)
            raise InvalidRun('hosted_billing_mode_invalid')
        await self.ledger.finish(reservation, status=status, elapsed_ms=elapsed_ms, usage=usage,
                                 outcome='completed' if 200 <= status < 300 else 'http_error', receipts=self.receipts)
        # aread() has already decoded gzip/deflate; do not decode it again.
        decoded_headers = response.headers.copy()
        decoded_headers.pop('content-encoding', None)
        decoded_headers['content-length'] = str(len(response_body))
        return httpx.Response(response.status_code, headers=decoded_headers, content=response_body,
                              extensions=response.extensions, request=guarded)

    async def aclose(self) -> None:
        await self.inner.aclose()


def _aggregate(outcomes, rows, include_verdicts=False) -> dict:
    """Aggregate completed outcomes. `None` entries are outcomes never produced (stopped run)."""
    completed = [outcome for outcome in outcomes if outcome is not None]
    models = {}
    for alias in MODEL_IDS:
        results = [outcome for outcome in completed if outcome[0] == alias]
        correct = sum(outcome[1] is True for outcome in results)
        models[alias] = {'correct': correct, 'total': len(results),
                         'accuracy': correct / len(results) if results else 0.0,
                         'parseFailures': sum(outcome[1] is None for outcome in results)}
    correct = sum(m['correct'] for m in models.values())
    result = {'score': correct / len(completed) if completed else 0,
              'metrics': {'models': models, 'outcomes': len(completed), 'questionCount': len(rows),
                          'parseFailures': sum(m['parseFailures'] for m in models.values()),
                          'tokensIn': sum(o[2] for o in completed), 'tokensOut': sum(o[3] for o in completed)}}
    if include_verdicts:
        # Question-major job order: outcomes[q * len(MODEL_IDS) + m] is question q, model m.
        marks = {True: '1', False: '0', None: 'u'}
        result['metrics']['verdicts'] = {
            alias: ''.join(marks[outcomes[q * len(MODEL_IDS) + m][1]] for q in range(len(rows)))
            for m, alias in enumerate(MODEL_IDS)}
    return result


async def _cancel_active(active) -> None:
    for task in active:
        task.cancel()
    await asyncio.gather(*active, return_exceptions=True)


async def evaluate(rows, template, complete, *, timeout=600, concurrency=6, drain_errors=False,
                   model_concurrency=None, record_correct=None, include_verdicts=False,
                   shuffle_seed=None, round_questions=None):
    if type(concurrency) is not int or not 1 <= concurrency <= 600:
        raise InvalidRun('invalid_concurrency')
    if model_concurrency is not None:
        if (set(model_concurrency) != set(MODEL_IDS)
                or any(type(limit) is not int or limit <= 0 for limit in model_concurrency.values())):
            raise InvalidRun('invalid_model_concurrency')
    staged = record_correct is not None or shuffle_seed is not None or round_questions is not None
    if staged and model_concurrency is None:
        raise InvalidRun('invalid_stop_policy')
    if ((record_correct is not None and (type(record_correct) is not int or record_correct < 0))
            or (round_questions is not None and (type(round_questions) is not int or round_questions < 1))
            or (shuffle_seed is not None and not isinstance(shuffle_seed, str))):
        raise InvalidRun('invalid_stop_policy')
    if include_verdicts and shuffle_seed is not None:
        # Verdict strings are indexed by the caller's row order; they exist only for the public panel.
        raise InvalidRun('invalid_verdict_policy')
    rows = list(rows)
    if shuffle_seed is not None:
        # Per-run order: a fixed private order would make the stopping point a prefix oracle on labels.
        random.Random(shuffle_seed).shuffle(rows)
    async def invoke(alias, prompt):
        # Admission queues are outside the per-attempt provider deadline.
        return await asyncio.wait_for(complete(alias, prompt), timeout)
    async def one(alias, row):
        prompt = render_prompt(template, row['equation1'], row['equation2'])
        response = await invoke(alias, prompt)
        if response.finish_reason not in ('stop', 'length', 'content_filter'):
            raise InvalidRun('provider_response_invalid')
        correct, _ = judge_response(response.text if not response.refusal else '', row['answer'])
        return alias, correct, response.tokens_in or 0, response.tokens_out or 0

    jobs = [(alias, row) for row in rows for alias in MODEL_IDS]
    if model_concurrency is None:
        semaphore = asyncio.Semaphore(concurrency)
        async def limited(alias, row):
            async with semaphore:
                return await one(alias, row)
        tasks = [asyncio.create_task(limited(alias, row)) for alias, row in jobs]
        try:
            outcomes = await asyncio.gather(*tasks, return_exceptions=drain_errors)
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    else:
        # Admit global and per-model capacity atomically. Nested semaphores can
        # queue one model ahead of another at the global boundary, leaving a
        # long capped-model tail even though eligible work remains.
        pending = {alias: deque() for alias in MODEL_IDS}
        round_size = (round_questions or len(rows)) * len(MODEL_IDS)
        rounds = deque(list(enumerate(jobs))[start:start + round_size] for start in range(0, len(jobs), round_size))
        rounds_total = len(rounds)

        def load_round():
            for index, (alias, row) in rounds.popleft():
                pending[alias].append((index, row))

        load_round()
        active_by_model = {alias: 0 for alias in MODEL_IDS}
        active = {}
        outcomes = [None] * len(jobs)
        completed = correct_so_far = rounds_cleared = 0
        stopped = False

        def admit():
            made_progress = True
            while len(active) < concurrency and made_progress:
                made_progress = False
                for alias in MODEL_IDS:
                    if len(active) >= concurrency:
                        break
                    if pending[alias] and active_by_model[alias] < model_concurrency[alias]:
                        index, row = pending[alias].popleft()
                        task = asyncio.create_task(one(alias, row))
                        active[task] = (index, alias)
                        active_by_model[alias] += 1
                        made_progress = True

        admit()
        try:
            while active:
                done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                failures = []
                for task in done:
                    index, alias = active.pop(task)
                    active_by_model[alias] -= 1
                    try:
                        outcomes[index] = task.result()
                        completed += 1
                        correct_so_far += outcomes[index][1] is True
                    except BaseException as exc:
                        outcomes[index] = exc
                        failures.append((index, exc))
                if failures and not drain_errors:
                    raise min(failures, key=lambda item: item[0])[1]
                if not active and not any(pending.values()) and not failures:
                    # Round boundary: nothing in flight. Stop only when even a perfect finish
                    # could not exceed the record; a finished panel is always a complete score.
                    rounds_cleared += 1
                    remaining = len(jobs) - completed
                    if record_correct is not None and remaining and correct_so_far + remaining <= record_correct:
                        stopped = True
                        break
                    if rounds:
                        load_round()
                admit()
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            if not stopped and (rounds or any(pending.values())):
                raise InvalidRun('missing_outcomes')
        except BaseException:
            await _cancel_active(active)
            raise
        if stopped:
            result = _aggregate(outcomes, rows)
            result['score'] = 0
            result['metrics']['stopped'] = {
                'roundsCleared': rounds_cleared, 'roundsTotal': rounds_total,
                'roundQuestions': round_questions or len(rows), 'completedOutcomes': completed,
                'correctSoFar': correct_so_far, 'recordCorrect': record_correct}
            return result
    if any(outcome is None for outcome in outcomes) or len(outcomes) != len(rows) * len(MODEL_IDS):
        raise InvalidRun('missing_outcomes')
    return _aggregate(outcomes, rows, include_verdicts)


@dataclass(frozen=True)
class ScreenPanel:
    rows: list
    sha256: str
    version: str
    min_correct: int
    max_parse_failures: int


def public_panel(screen: ScreenPanel, metrics: dict) -> dict:
    """The public-panel block: verbatim public problems, per-model verdicts, counts and the floor."""
    models = {alias: dict(metrics['models'][alias]) for alias in MODEL_IDS}
    correct = sum(m['correct'] for m in models.values())
    parse_failures = sum(m['parseFailures'] for m in models.values())
    return {
        'datasetKind': 'public', 'datasetVersion': screen.version, 'datasetSha256': screen.sha256,
        'questionCount': len(screen.rows),
        'problems': [{'id': row['id'], 'eq1Id': row['eq1_id'], 'eq2Id': row['eq2_id'],
                      'equation1': row['equation1'], 'equation2': row['equation2'], 'expected': row['answer']}
                     for row in screen.rows],
        'verdicts': dict(metrics['verdicts']),
        'models': models,
        'correct': correct, 'total': sum(m['total'] for m in models.values()), 'parseFailures': parse_failures,
        'tokensIn': metrics['tokensIn'], 'tokensOut': metrics['tokensOut'],
        'floor': {'minCorrect': screen.min_correct, 'maxParseFailures': screen.max_parse_failures,
                  'passed': correct >= screen.min_correct and parse_failures <= screen.max_parse_failures},
    }


async def live_evaluate(rows, template, env, *, screen: ScreenPanel | None = None, record_correct=None,
                        shuffle_seed=None, round_questions=None, include_verdicts=False):
    key, cap = live_settings(env)
    uncapped = uncapped_hosted_mode(env)
    configs = load_models(VENDOR / 'evaluation_models.json')
    if set(configs) != set(MODEL_IDS):
        raise InvalidRun('model_config_mismatch')
    ledger = BudgetLedger(cap)
    hosted = hosted_workflow(env)
    completed = {alias: 0 for alias in MODEL_IDS}
    receipts = private_receipts(env)
    transport = GuardedTransport(base_transport(), ledger, receipts, reject_byok=uncapped)
    status = 'failed'
    try:
        async with httpx.AsyncClient(transport=transport, timeout=180, trust_env=False, follow_redirects=False) as client:
            response = await client.get('https://openrouter.ai/api/v1/key', headers={'Authorization': f'Bearer {key}'})
            response.raise_for_status()
            if not uncapped:
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
                completed[alias] += 1
                if hosted and completed[alias] % 20 == 0:
                    print(json.dumps({'event': 'evaluation_progress', 'model': alias,
                                      'completed': completed[alias]}), flush=True)
                return response
            panel = None
            if screen is not None:
                # Stage one: the public panel, same ledger and client, before any private spend.
                screened = await evaluate(screen.rows, template, complete, model_concurrency=MODEL_CONCURRENCY,
                                          include_verdicts=True)
                panel = public_panel(screen, screened['metrics'])
                for alias in MODEL_IDS:
                    completed[alias] = 0
                if not panel['floor']['passed']:
                    result = _aggregate([], rows)
                    result['score'] = 0
                    result['metrics'].update({'stage': 'screen_failed', 'partial': True, 'publicPanel': panel,
                                              'billing': ledger.metrics()})
                    status = 'screen_failed'
                    return result
            result = await evaluate(rows, template, complete, model_concurrency=MODEL_CONCURRENCY,
                                    record_correct=record_correct, shuffle_seed=shuffle_seed,
                                    round_questions=round_questions, include_verdicts=include_verdicts)
            if panel is not None:
                result['metrics']['publicPanel'] = panel
                if 'stopped' in result['metrics']:
                    result['metrics'].update({'stage': 'ranked_stopped', 'partial': True})
                    status = 'stopped'
                else:
                    result['metrics']['stage'] = 'ranked_complete'
                    status = 'succeeded'
            else:
                status = 'succeeded'
            result['metrics']['billing'] = ledger.metrics()
            return result
    finally:
        if receipts is not None:
            receipts.close()
        if hosted:
            try:
                write_cost_artifact(ledger, status)
            except Exception:
                # Preserve the original evaluation failure; successful runs require observability.
                if status in ('succeeded', 'stopped', 'screen_failed'):
                    # Every publishing run is billed; a missing cost artifact must not publish silently.
                    raise InvalidRun('cost_artifact_unavailable') from None


async def run(args, env=os.environ, evaluator=live_evaluate):
    score_path = ROOT / 'score.json'
    score_path.unlink(missing_ok=True)
    (ROOT / 'evaluation-costs.json').unlink(missing_ok=True)
    (ROOT / 'evaluation-costs.json.tmp').unlink(missing_ok=True)
    verify_vendor()
    template, data = read_prompt(ROOT / 'submission/prompt.txt')
    ranked = args.ranked
    if ranked:
        dataset = Path(env.get('DISTILL_PRIVATE_DATASET', ''))
        if not dataset.is_absolute() or dataset.resolve().is_relative_to(ROOT):
            raise InvalidRun('private_dataset_must_be_external')
    else:
        dataset = VENDOR / 'examples/problems_hard3_20.jsonl'
    rows, dataset_hash, version = load_dataset(dataset, ranked=ranked, manifest_path=ROOT / 'ranked-dataset.json')
    live_settings(env)  # Offline mode is tests, never a fake publishable score.
    candidate = env.get('GITHUB_SHA', '')
    if ranked and not re.fullmatch('[0-9a-f]{40}', candidate):
        raise InvalidRun('candidate_identity_missing')
    screen_rows, screen_hash, screen_version = load_dataset(SCREEN_DATASET, ranked=False)
    min_correct, max_parse_failures = screen_policy(env)
    stop_enabled, record_url, round_questions = stop_policy(env)
    screen = ScreenPanel(screen_rows, screen_hash, screen_version, min_correct, max_parse_failures)
    record = None
    # Always announce the record decision so a misconfigured environment is visible in the log.
    if not ranked:
        reason = 'public_run'
    elif not stop_enabled:
        reason = 'disabled'
    elif not record_url:
        reason = 'url_unset'
    else:
        record = await fetch_record(record_url)
        reason = 'fetched' if record is not None else 'unavailable'
    print(json.dumps({'event': 'record', 'recordCorrect': None if record is None else record['correct'],
                      'reason': reason}), flush=True)
    if ranked:
        assert_screen_disjoint(screen_rows, rows)
        result = await evaluator(rows, template, env, screen=screen,
                                 record_correct=None if record is None else record['correct'],
                                 shuffle_seed=private_shuffle_seed(), round_questions=round_questions)
    else:
        result = await evaluator(rows, template, env, include_verdicts=True)
        panel = public_panel(screen, result['metrics'])
        result['metrics'].pop('verdicts', None)
        result['metrics'].update({'stage': 'public_complete', 'publicPanel': panel})
    policy_on = stop_enabled and record_url is not None
    result['metrics']['evaluationPolicy'] = {
        'screen': {'panel': screen_version, 'minCorrect': min_correct, 'maxParseFailures': max_parse_failures},
        'stopWhenImpossible': ranked and policy_on, 'roundQuestions': round_questions,
        'record': record if record is not None else ('unavailable' if ranked and policy_on else 'disabled'),
        'order': 'shuffled-question-major', 'shuffleSeed': 'private-random'}
    result['metrics'].update({'contractVersion': CONTRACT, 'datasetKind': 'ranked' if ranked else 'public',
                              'datasetVersion': version, 'datasetSha256': dataset_hash,
                              'configSha256': digest((VENDOR / 'evaluation_models.json').read_bytes()),
                              'executionConfigSha256': digest(json.dumps(EXECUTION_CONFIG, sort_keys=True, separators=(',', ':')).encode()),
                              'promptBytes': len(data), 'promptSha256': digest(data),
                              'candidateSha': candidate, 'runId': env.get('GITHUB_RUN_ID', 'local'),
                              'runAttempt': env.get('GITHUB_RUN_ATTEMPT', 'local')})
    if result['metrics'].get('stage') not in STAGES:
        raise InvalidRun('stage_invalid')
    # Atomic publication; only this allowlisted aggregate object is exported.
    temporary = score_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, allow_nan=False, sort_keys=True) + '\n')
    temporary.replace(score_path)
    print(json.dumps({'score': result['score'], 'datasetKind': result['metrics']['datasetKind'],
                      'stage': result['metrics']['stage']}))


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
