"""Read authenticated original MiniCPM5 result JSON only; no raw tensor IO.

Observe with the existing shared shell gate. This is a metadata diagnosis, not
a new native/Neuron/saved-tensor job or a runtime admission.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from minicpm5_saved_tap_analysis_v1 import ARMS, TAPS, STAGES, compare_call, validate_call, require

R = Path(__file__).resolve().parents[1]
Q = R/'research'
BASE = Path('/mnt/nvme/minicpm5-original32-layer0-full-bf16-cpu-r1-20261108-v1')
EXIT = dict(path=str(Q/'minicpm5-layer0-full-bf16-cpu-root-exit-r1-20261108-v1.json'),
    sha256='727d92a3189bc671016701e53bfc982547c33917cb046dfd2ca29eb78f84e0ff')
OUT = Q/'minicpm5-saved-tap-metadata-diagnosis-20261109-v1.json'


class Reader:
    def __init__(self):
        self.pins, self.bytes = {}, 0

    def __call__(self, ref):
        require(set(ref) == {'path','sha256'}, 'Literal metadata reference')
        path = Path(ref['path'])
        require(path.suffix == '.json' and path.resolve() == path
                and (path.is_relative_to(Q) or path.is_relative_to(BASE))
                and not any(p.is_symlink() for p in (path,*path.parents)), 'Only fixed ordinary JSON roots')
        fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 8*2**20, 'Bounded JSON')
            with os.fdopen(fd,'rb',closefd=False) as stream:
                raw = stream.read(8*2**20+1)
            after = os.fstat(fd)
            current = path.stat()
            require(len(raw) == before.st_size and all(getattr(before,k) == getattr(after,k) == getattr(current,k)
                for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')), 'Stable metadata')
        finally:
            os.close(fd)
        digest = hashlib.sha256(raw).hexdigest()
        require(digest == ref['sha256'] and (str(path) not in self.pins or self.pins[str(path)] == digest),
                'Original metadata fingerprint')
        self.pins[str(path)] = digest
        self.bytes += len(raw)
        require(self.bytes <= 64*2**20, 'Bounded complete metadata reads')
        return json.loads(raw)


def analyze(read):
    ended = read(EXIT)
    require(ended['actual_exit_observed'] is True and ended['actual_exit_code'] == 0
            and ended['all_four_processes_gone'] is True and ended['cgroup_empty'] is True,
            'Original actually completed native trajectory')
    spec = read(ended['spec'])
    require(spec['output_root'] == str(BASE), 'Exact original output namespace')
    plan = read(spec['plan'])
    require(plan['output_root'] == str(BASE) and len(plan['frames']) == 32, 'Full original trajectory')
    ranks, calls = [], {}
    bundle_inventory = []
    for rank in (0,1):
        rr = read(ended['outputs'][f'rank{rank}_result'])
        require(rr['complete'] is True and rr['rank'] == rank and rr['arms'] == list(ARMS)
                and rr['model_forwards'] == 42 and rr['selected_tap_names'] == list(TAPS),
                'Original complete three-arm rank')
        require([r['arm'] for r in rr['arm_reports']] == list(ARMS), 'All original arms ordered')
        ranks.append(rr)
        for item,arm in zip(rr['arm_reports'],ARMS):
            expected = BASE/f'cpu-rank{rank}'/arm/'result.json'
            require(item['report'] == str(expected), 'Exact original arm record')
            report = read(dict(path=item['report'],sha256=item['sha256']))
            require(report['complete'] is True and report['arm'] == arm and report['rank'] == rank
                    and report['model_forwards'] == 14
                    and [r['call'] for r in report['reports']] == list(range(14)),
                    'All14 original calls')
            rows = []
            for ci,entry in enumerate(report['reports']):
                require(entry['report'] == str(BASE/f'cpu-rank{rank}'/arm/f'call-{ci:02d}'/'result.json'),
                        'Exact call namespace')
                call = read(dict(path=entry['report'],sha256=entry['sha256']))
                validate_call(call)
                require(call['call']['index'] == ci and call['arm'] == arm and call['rank'] == rank,
                        'Original call identity')
                bundle = call['selected_tap_bundle']
                require(bundle['path'] == str(Path(entry['report']).parent/'selected-taps.bf16'),
                        'Exact selected-tap path')
                bundle_inventory.append(dict(rank=rank,arm=arm,call=ci,bundle=bundle))
                rows.append(call)
            require([f for c in rows for f in c['call']['frames']] == plan['frames'],
                    'All32 original frames without cherry-picking')
            calls[rank,arm] = rows
    require(ranks[0]['summaries'] == ranks[1]['summaries'], 'Replicated rank endpoint summaries')
    for arm in ARMS:
        for left,right in zip(calls[0,arm],calls[1,arm]):
            require(left['stage_names'] == right['stage_names']
                    and left['stage_sha256'] == right['stage_sha256']
                    and left['HF_metrics'] == right['HF_metrics']
                    and left['output_metadata'] == right['output_metadata'],
                    'All213 replicated prediction hashes and final endpoints agree across ranks')
            for key in ('sha256','encoding','raw_bytes','entries'):
                require(left['selected_tap_bundle'][key] == right['selected_tap_bundle'][key],
                        'Cross-rank selected bundle descriptors match')
    comparisons = []
    for a,b in ((ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[1],ARMS[2])):
        rows = [compare_call(x,y) for x,y in zip(calls[0,a],calls[0,b])]
        comparisons.append(dict(left_arm=a,right_arm=b,rows=rows,
            calls_with_different_logits=sum(not r['logits_hash_equal'] for r in rows),
            first_difference_counts={name:sum(r['first_recorded_different_stage'] == name for r in rows)
                for name in (None,*STAGES) if any(r['first_recorded_different_stage'] == name for r in rows)}))
    require(len(bundle_inventory) == 84 and sum(r['bundle']['raw_bytes'] for r in bundle_inventory) == 3145728,
            'Full original84 bundles and all3145728 declared bytes')
    return dict(schema='minicpm5_saved_tap_metadata_diagnosis_v1',complete=True,
        original_actual_exit=EXIT,original_spec=ended['spec'],original_plan=spec['plan'],
        rank_count=2,ranks_counted_once=True,arms=list(ARMS),calls_per_arm=14,frames_per_arm=32,
        prediction_stages_per_call=213,comparisons=comparisons,bundle_inventory=bundle_inventory,
        original_summaries=ranks[0]['summaries'],original_remaining_gates=plan['remaining_gates'],
        tensor_payloads_opened=False,weight_payloads_opened=False,
        model_reexecuted=False,numeric_tap_statistics_computed=False,causal_root_cause_established=False,
        full_model_qualified=False,new_speedup_claim=False,
        limitations=[
          'Stage hashes describe complete prediction-row batches; a four-row decode difference cannot be assigned to an individual frame from a hash.',
          'Only layer0 attention/output and layer41 attention/output have saved selected numerical tensors in this run; the other209 stages have metadata/hashes only.',
          'Hash inequality does not quantify error, sign, amplification, HF closeness, or causality.',
          'Anchor itself fails3/32 CPU frames; candidate-versus-anchor deltas cannot be treated as absolute HF errors.',
          'The former original FP32 metrics remain authoritative; future float64 descriptive tap statistics do not change thresholds.',
          'Future raw-byte analysis must authenticate all84 bundles, preserve physical-row permutations, verify both ranks, and run in a separate serial admitted window.'
        ])


def main():
    require(sys.dont_write_bytecode and __debug__, 'Original assertions and no bytecode writes')
    read = Reader()
    result = analyze(read)
    result.update(source_sha256={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        str(Q/'minicpm5_saved_tap_analysis_v1.py'):hashlib.sha256((Q/'minicpm5_saved_tap_analysis_v1.py').read_bytes()).hexdigest()},
        metadata_source_sha256=read.pins,metadata_bytes_read=read.bytes)
    raw = (json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
    require(len(raw) <= 4*2**20, 'Bounded metadata diagnosis')
    fd = os.open(OUT,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(result=dict(path=str(OUT),sha256=hashlib.sha256(raw).hexdigest()),
        metadata_files=len(read.pins),metadata_bytes=read.bytes,
        comparisons=[{k:v for k,v in x.items() if k != 'rows'} for x in result['comparisons']],
        tensor_payloads_opened=False)))


if __name__ == '__main__':
    main()
