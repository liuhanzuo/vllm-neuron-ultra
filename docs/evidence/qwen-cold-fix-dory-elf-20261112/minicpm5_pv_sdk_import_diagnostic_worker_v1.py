"""Only original SDK phase1; inert unless explicitly released by its parent."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'research'), str(ROOT)]
from research.minicpm5_pv_sdk_import_diagnostic_v1 import worker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', required=True)
    parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--phase', type=int, choices=(1,), required=True)
    args = parser.parse_args()
    sys.exit(worker(dict(path=args.admission, sha256=args.admission_sha256), args.phase, str(Path(__file__).resolve())))
