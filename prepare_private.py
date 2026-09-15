"""Decode the operator-provided compressed fixture into runner-owned temporary space."""
import base64
import gzip
import os
from pathlib import Path
from evaluate import load_dataset, ROOT


def main():
    target = Path(os.environ['DISTILL_PRIVATE_DATASET'])
    if not target.is_absolute() or target.resolve().is_relative_to(ROOT):
        raise ValueError('private destination must be external')
    compressed = base64.b64decode(os.environ['DISTILL_PRIVATE_DATA_GZIP_BASE64'], validate=True)
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        data = stream.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError('private dataset too large')
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
        stream.write(data)
    load_dataset(target, ranked=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        raise SystemExit('Private dataset unavailable or invalid; no score produced.') from None
