"""Operator-only selection. No model calls; private outputs must be outside this repo."""
from __future__ import annotations
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import urllib.request

SOURCE_SHA = '1aec8a7acf223b7c56e4830977b6e90d4ef1924b'
SAIR_SHA = 'cf2e964ae911e21421bc9dbf7e28cc8df7291983'
CONTRACT_VERSION = 'distill-v2'
DATASET_VERSION = 'yukon-equational-v1'
ROOT = Path(__file__).resolve().parent


def number(name):
    if not isinstance(name, str) or not re.fullmatch(r'Equation[0-9]+', name):
        raise ValueError('invalid equation identity')
    return int(name[8:])


def candidates(entries, excluded):
    positives, negatives = {}, {}
    for entry in entries:
        if entry.get('proven') is not True:
            continue
        variant = entry['variant']
        provenance = {k: entry[k] for k in ('name', 'filename', 'line')}
        implication = variant.get('implication')
        if implication and implication['finite'] is False:
            pair = (number(implication['lhs']), number(implication['rhs']))
            positives.setdefault(pair, provenance)
        facts = variant.get('facts')
        if facts:
            # A finite counterexample also disproves an implication over all magmas.
            for lhs in facts['satisfied']:
                for rhs in facts['refuted']:
                    negatives.setdefault((number(lhs), number(rhs)), provenance)
    conflicts = positives.keys() & negatives.keys()
    def eligible(pool):
        return {pair: provenance for pair, provenance in pool.items()
                if pair not in excluded and pair not in conflicts
                and pair[0] != pair[1] and all(1 <= n <= 4694 for n in pair)}
    return eligible(positives), eligible(negatives)


def select(entries, excluded, equations, seed):
    if len(seed) != 32:
        raise ValueError('seed must be 32 bytes')
    positive, negative = candidates(entries, excluded)
    rows = []
    for answer, pool in ((True, positive), (False, negative)):
        if len(pool) < 100:
            raise ValueError('insufficient eligible proven pairs')
        def ordering(pair):
            return hmac.new(seed, f'{int(answer)}:{pair[0]}:{pair[1]}'.encode(), hashlib.sha256).digest()
        for pair in sorted(pool, key=ordering)[:100]:
            rows.append({'id': f'yukon_{len(rows)+1:04}', 'eq1_id': pair[0], 'eq2_id': pair[1],
                         'equation1': equations[pair[0]], 'equation2': equations[pair[1]],
                         'answer': answer, 'provenance': {**pool[pair], 'sourceSha': SOURCE_SHA}})
    # Interleave classes, avoiding a class-dependent request order across all models.
    return [row for pair in zip(rows[:100], rows[100:]) for row in pair]


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def sources():
    base = f'https://raw.githubusercontent.com/teorth/equational_theories/{SOURCE_SHA}/'
    entries_raw = fetch(base + 'full_entries.json')
    equations_raw = fetch(base + 'data/equations.txt')
    lines = equations_raw.decode().splitlines()
    if len(lines) != 4694 or any(' = ' not in line for line in lines):
        raise ValueError('unrecognized equation source format')
    equations = {i: line.replace('◇', '*') for i, line in enumerate(lines, 1)}
    index = json.loads(fetch(f'https://huggingface.co/api/datasets/SAIRfoundation/equational-theories-selected-problems/revision/{SAIR_SHA}'))
    if index.get('sha') != SAIR_SHA:
        raise ValueError('dataset revision mismatch')
    paths = sorted(x['rfilename'] for x in index['siblings'] if x['rfilename'].startswith('data/') and x['rfilename'].endswith('.jsonl'))
    if not paths or not any('evaluation' in p for p in paths):
        raise ValueError('published exclusion inventory incomplete')
    checksums = {'full_entries.json': hashlib.sha256(entries_raw).hexdigest(), 'equations.txt': hashlib.sha256(equations_raw).hexdigest()}
    excluded = set()
    for path in paths:
        data = fetch(f'https://huggingface.co/datasets/SAIRfoundation/equational-theories-selected-problems/resolve/{SAIR_SHA}/{path}')
        checksums[path] = hashlib.sha256(data).hexdigest()
        for line in data.splitlines():
            row = json.loads(line)
            if any(type(row.get(key)) is not int for key in ('eq1_id', 'eq2_id')):
                raise ValueError('published pair cannot be identified')
            excluded.add((row['eq1_id'], row['eq2_id']))
    return json.loads(entries_raw), excluded, equations, checksums


def external(path):
    resolved = path.resolve()
    # This bundle is kept inside Yukon until handoff. Protect both trees.
    roots = [ROOT, *[p for p in ROOT.parents if (p / '.git').exists()]]
    if any(resolved.is_relative_to(root) for root in roots):
        raise ValueError('private outputs must be outside every repository')
    return resolved


def write_private(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed-file', type=Path, help='Reuse an existing private 32-byte seed')
    args = parser.parse_args()
    target = external(args.output_dir)
    if target.exists():
        raise ValueError('output directory must be new; never overwrite a frozen selection')
    seed = external(args.seed_file).read_bytes() if args.seed_file else secrets.token_bytes(32)
    entries, excluded, equations, checksums = sources()
    rows = select(entries, excluded, equations, seed)
    data = b''.join((json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n').encode() for row in rows)
    target.mkdir(parents=True, mode=0o700)
    write_private(target / 'selection.seed', seed)
    write_private(target / 'ranked.jsonl', data)
    write_private(target / 'provenance.json', json.dumps({'sourceSha': SOURCE_SHA, 'excludedDatasetSha': SAIR_SHA, 'checksums': checksums, 'excludedPairs': len(excluded)}, indent=2).encode())
    manifest = {'contractVersion': CONTRACT_VERSION, 'version': DATASET_VERSION, 'sha256': hashlib.sha256(data).hexdigest()}
    write_private(target / 'ranked-dataset.json', (json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps(manifest))  # Only public identity; never print the selection or seed.


if __name__ == '__main__':
    main()
