"""Explicit future diagnostic spec/review CLI; inert on import."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'research'), str(ROOT)]
from research.minicpm5_pv_sdk_import_diagnostic_v1 import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('spec', 'review'):
        parser.add_argument('--'+name, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    args = parser.parse_args()
    result = run(dict(path=args.spec, sha256=args.spec_sha256), dict(path=args.review, sha256=args.review_sha256))
    print(json.dumps(result), flush=True)
    code = result['terminal']['returncode']
    # Preserve all actual child returncodes in the durable parent receipt;
    # a child killed by signal produces a nonzero outer process status.
    sys.exit(code if code >= 0 else 128-code)
