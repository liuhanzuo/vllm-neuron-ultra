"""Finite original-interpreter PV worker, inert unless explicitly launched."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'research'), str(ROOT)]
from research.minicpm5_layer1_fixed_input_pv_adapter_v2 import worker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', required=True)
    parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--phase', type=int, choices=range(5), required=True)
    args = parser.parse_args()
    worker(dict(path=args.admission, sha256=args.admission_sha256), args.phase, str(Path(__file__).resolve()))
