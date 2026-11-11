"""Reviewed finite PV outer entry. Does nothing unless explicitly launched."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'research'), str(ROOT)]
from research.minicpm5_layer1_fixed_input_pv_adapter_v2 import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('spec', 'review'):
        parser.add_argument('--'+name, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    args = parser.parse_args()
    result = run(dict(path=args.spec, sha256=args.spec_sha256), dict(path=args.review, sha256=args.review_sha256))
    print(json.dumps(result), flush=True)
