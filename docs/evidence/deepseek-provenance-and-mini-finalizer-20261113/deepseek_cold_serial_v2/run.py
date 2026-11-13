"""Inert CLI until explicit bindings; outer currently rejects missing layout proof."""
import argparse
from pathlib import Path
import sys
Q=Path(__file__).resolve().parent.parent
for directory in(Q,Q.parent):
    if str(directory)not in sys.path:sys.path.insert(0,str(directory))

def main(argv=None):
    from deepseek_cold_serial_v2.contract import reference,require
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('outer','coordinator','supervisor','leaf'),default='outer')
    for name in('spec','spec-sha256','review','review-sha256','binding','binding-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args(argv)
    require(sys.dont_write_bytecode and __debug__,'Explicit -B and assertions required')
    if args.mode=='outer':
        require(all((args.spec,args.spec_sha256,args.review,args.review_sha256))and not args.binding and not args.binding_sha256,
            'Explicit future independently reviewed complete spec/review only')
        from deepseek_cold_serial_v2.runtime import outer
        return outer(reference(dict(path=args.spec,sha256=args.spec_sha256)),reference(dict(path=args.review,sha256=args.review_sha256)))
    require(args.binding and args.binding_sha256 and not any((args.spec,args.spec_sha256,args.review,args.review_sha256)),
        'Only exact admitted worker binding')
    binding=reference(dict(path=args.binding,sha256=args.binding_sha256))
    from deepseek_cold_serial_v2.bootstrap import coordinator,native_leaf,worker_bootstrap
    if args.mode=='coordinator':return coordinator(binding)
    if args.mode=='leaf':return native_leaf(binding)
    from deepseek_cold_serial_v2.service import supervise
    return supervise(binding,worker_bootstrap)

if __name__=='__main__':raise SystemExit(main())
