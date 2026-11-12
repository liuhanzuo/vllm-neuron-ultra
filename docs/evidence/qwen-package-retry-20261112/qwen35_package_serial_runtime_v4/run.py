#!/usr/bin/env python3
"""Explicit future serial sampler CLI; --help/import never execute work."""
import argparse
from pathlib import Path
import sys


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("outer", "coordinator", "supervisor", "leaf", "api_bootstrap"), default="outer")
    value.add_argument("--spec"); value.add_argument("--spec-sha256")
    value.add_argument("--review"); value.add_argument("--review-sha256")
    value.add_argument("--binding"); value.add_argument("--binding-sha256")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    # Exact experiment and research paths in every interpreter, including
    # fresh systemd spawn; this adds no native imports or defaults/bindings.
    for path in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    from qwen35_package_serial_runtime_v4.contract import reference, require
    require(sys.dont_write_bytecode and __debug__, "-B and unoptimized explicit interpreter required")
    if args.mode == "outer":
        require(all((args.spec, args.spec_sha256, args.review, args.review_sha256)) and args.binding is None
                and args.binding_sha256 is None, "explicit future reviewed spec/review refs required")
        spec = reference({"path": args.spec, "sha256": args.spec_sha256})
        review = reference({"path": args.review, "sha256": args.review_sha256})
        from qwen35_package_serial_runtime_v4.adapter import outer
        return outer(spec, review)
    require(args.binding and args.binding_sha256 and not any((args.spec, args.spec_sha256, args.review, args.review_sha256)),
            "exact authenticated child binding only")
    binding = reference({"path": args.binding, "sha256": args.binding_sha256})
    from qwen35_package_serial_runtime_v4.adapter import coordinator, native_leaf, worker_bootstrap
    if args.mode == "coordinator": return coordinator(binding)
    if args.mode == "supervisor":
        from qwen35_package_serial_runtime_v4.service import supervise
        return supervise(binding, worker_bootstrap)
    return native_leaf(binding, api=args.mode == "api_bootstrap")


if __name__ == "__main__":
    raise SystemExit(main())
