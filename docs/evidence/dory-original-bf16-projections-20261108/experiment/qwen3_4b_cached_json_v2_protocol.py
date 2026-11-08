"""Keep the original Qwen3-4B long protocol while changing only its HF oracle."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'qwen3_4b-cached-baseline-json-v2'
REFERENCE = ROOT / 'qwen3_4b-cached-reference-json-v2'
STEPS = 2041
MARKS = [0,22,30,31,32,33,63,64,65,126,127,128,129,255,256,257,
         510,511,512,513,1022,1023,1024,1025,2042,2043,2047,2048,2049,
         2168,2169,3063,3064,3065]
CRITERIA = {'cosine_min': .995, 'kl_max_nats': .02, 'max_abs_error_max': 2.,
            'argmax_match_fraction_min': .9, 'all_reference_margins_gt_0.5_must_match': True}
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build():
    native = json.loads((ROOT / 'qwen3_4b/long-decode-hf-reference.json').read_text())
    assert native['revision'] == REVISION and native['criteria'] == CRITERIA
    assert native['sequence_length'] == 3072 and native['positions'] == MARKS
    fixtures = native['fixtures']
    assert [row['id'] for row in fixtures] == [0, 1]
    assert all(len(row['input_ids']) == 3072 for row in fixtures)
    trajectories = [dict(id=i, fixture=f, prefix=p, slot=s,
                         last_position=p+STEPS-1+(2 if s != 3 else 0))
                    for i, (f,p,s) in enumerate([(0,23,3),(1,127,1),(0,1023,0),(1,1,2)])]
    trajectories.append(dict(id=4, fixture=1, prefix=31, slot=3, last_position=32))
    operations = []
    def req(t, pos, record=True):
        return dict(trajectory=t['id'], fixture=t['fixture'], slot=t['slot'], position=pos, record=record)
    for t in trajectories[:4]:
        operations.append(dict(phase='prefill', requests=[req(t,t['prefix']-1)]))
    for step in range(STEPS):
        operations.append(dict(phase='long_decode', requests=[req(t,t['prefix']+step,t['prefix']+step in MARKS)
                                                              for t in trajectories[:4]]))
    operations += [dict(phase='slot_reuse_prefill', requests=[req(trajectories[4],30)]),
                   dict(phase='decode_after_slot_reuse', requests=[req(trajectories[4],31)] +
                        [req(t,t['prefix']+STEPS) for t in trajectories[1:4]]),
                   dict(phase='partial_batch', requests=[req(t,t['prefix']+STEPS+1) for t in trajectories[1:4]]),
                   dict(phase='resume_inactive', requests=[req(trajectories[4],32)])]
    checks = [{k: req[k] for k in ('trajectory','fixture','position','slot')} | {'phase': op['phase']}
              for op in operations for req in op['requests'] if req['record']]
    return dict(fixtures=fixtures, trajectories=trajectories, operations=operations, checks=checks,
                criteria=CRITERIA, revision=REVISION, native_source_sha256=native['reference_source_sha256'])


def calls_for(protocol):
    result = []
    for op in protocol['operations']:
        prefill = op['phase'] in ('prefill','slot_reuse_prefill')
        requests = op['requests']
        if prefill:
            r = requests[0]; n = r['position']+1
            ids = [protocol['fixtures'][r['fixture']]['input_ids'][:n]]
            positions = [list(range(n))]
        else:
            ids = [[protocol['fixtures'][r['fixture']]['input_ids'][r['position']]] for r in requests]
            positions = [[r['position']] for r in requests]
        result.append(dict(ids=ids, positions=positions, slots=[r['slot'] for r in requests], prefill=prefill))
    return result


def original_trace(fixtures):
    """Execute only the old loop AST with arrays and recording stubs, never Torch."""
    source = ROOT / 'validate_qwen3_4b_long_decode.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='neuron')
    loop = next(node for node in function.body if isinstance(node,ast.With))
    calls, checks = [], []
    class Array:
        def __init__(self, data): self.data = data
        def __getitem__(self, index):
            if index == (slice(None), None): return Array([[x] for x in self.data])
            return Array([self.data]) if index is None else None
    class Arrays:
        tensor = staticmethod(lambda value: Array(value))
        arange = staticmethod(lambda value: Array(list(range(value))))
    def forward(ids, positions, slots, prefill=False):
        calls.append(dict(ids=ids.data, positions=positions.data, slots=slots.data, prefill=prefill))
        return Array(None)
    def record(logits, ref, pos, slot, phase):
        checks.append(dict(fixture=ref,position=pos,slot=slot,phase=phase))
    env = dict(torch=Arrays, forward=forward, record=record, evidence=fixtures,
               refs=[0,1,0,1], lengths=[23,127,1023,1], slots=[3,1,0,2],
               STEPS=STEPS, MARKS=MARKS, time=time, start=time.monotonic(), json=json,
               print=lambda *args,**kwargs: None)
    module = ast.fix_missing_locations(ast.Module(body=loop.body, type_ignores=[]))
    exec(compile(module,str(source),'exec'),env)
    return calls, checks


def validate(protocol):
    old_calls, old_checks = original_trace(protocol['fixtures'])
    assert calls_for(protocol) == old_calls, 'Forward tokens/positions/slots differ from original protocol'
    assert [{k:r[k] for k in ('fixture','position','slot','phase')} for r in protocol['checks']] == old_checks
    assert len(protocol['operations']) == STEPS+8 and len(protocol['checks']) == 93
    assert len({(r['trajectory'],r['position']) for r in protocol['checks']}) == 93
    live, counts = {}, {}
    for op in protocol['operations']:
        assert len({r['slot'] for r in op['requests']}) == len(op['requests'])
        for r in op['requests']:
            slot, tid, pos = r['slot'],r['trajectory'],r['position']
            if op['phase'] not in ('prefill','slot_reuse_prefill'):
                assert live[slot] == (tid,pos), 'Broken cache continuity'
            live[slot] = tid,pos+1
            assert 0 <= pos < 3072
            counts[tid] = counts.get(tid,0)+1
    for t in protocol['trajectories']:
        assert counts[t['id']] == t['last_position']-t['prefix']+2
    assert protocol['criteria'] == CRITERIA


def passed(rows):
    if not rows: return False
    names = ['cosine','kl_reference_to_neuron','max_abs_error','reference_margin']
    return all(all(math.isfinite(r[n]) for n in names) and r['cosine'] >= .995 and
               r['kl_reference_to_neuron'] <= .02 and r['max_abs_error'] <= 2. and
               (r['reference_margin'] <= .5 or r['argmax_match']) for r in rows) and \
           sum(r['argmax_match'] for r in rows)/len(rows) >= .9


def preflight():
    protocol = build(); validate(protocol)
    controls = []
    for label, change in [
        ('Dropped decode step rejected', lambda p:p['operations'].pop(100)),
        ('Changed token position rejected', lambda p:p['operations'][100]['requests'][0].update(position=0)),
        ('Changed slot rejected', lambda p:p['operations'][100]['requests'][0].update(slot=2)),
        ('Relaxed threshold rejected', lambda p:p['criteria'].update(max_abs_error_max=3.)),
    ]:
        modified = copy.deepcopy(protocol); change(modified)
        try: validate(modified)
        except AssertionError: controls.append(label)
        else: raise AssertionError(label)
    good=dict(cosine=1.,kl_reference_to_neuron=0.,max_abs_error=0.,reference_margin=1.,argmax_match=True)
    assert passed([good]) and not passed([{**good,'max_abs_error':2.1875}])
    assert not passed([{**good,'cosine':float('nan')}]) and not passed([])
    controls += ['Original 2.1875 error rejected','NaN and empty metrics rejected']
    proof = dict(passed=True, operations=len(protocol['operations']), checkpoints=len(protocol['checks']),
                 teacher_forced_decode_steps=STEPS, native_cached_steps=sum(t['last_position']-t['prefix']+1
                     for t in protocol['trajectories']), original_loop_tokens_positions_slots_exact=True,
                 trajectories=protocol['trajectories'], checks=protocol['checks'], criteria=CRITERIA,
                 operations_sha256=digest(protocol['operations']),
                 fixture_ids_sha256=digest([r['input_ids'] for r in protocol['fixtures']]), controls=controls,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope='Same complete original2041-step/93-frame protocol, now five native cached trajectories. '
                       'AST replay checks every original forward input. Original full-causal failure remains.')
    REFERENCE.mkdir(exist_ok=True)
    path=REFERENCE/'protocol-preflight.json'
    if path.exists(): assert json.loads(path.read_text()) == proof
    else: path.write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__ == '__main__':
    p=preflight(); print(json.dumps({k:p[k] for k in ['passed','operations','checkpoints','native_cached_steps','controls']}))
