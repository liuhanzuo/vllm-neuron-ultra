"""Source-only exact attention AST comparison; not compiled/device equivalence."""
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path
ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=ROOT/'research'
TARGET=Path('/mnt/nvme/smollm3-cache-ablation-native-v3/target/vllm_neuron/models/smollm3')
PINNED=Path('/home/ubuntu/src/vllm-neuron-ultra-smollm3-cache-ablation-v1/vllm_neuron/models/smollm3')

def dump(value):return ast.dump(value,include_attributes=False)
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
def expression(text):return ast.parse(text,mode='eval').body

class LowerOnlyCacheHelpers(ast.NodeTransformer):
    def __init__(self):self.reads=0;self.writes=0
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)and node.func.value.id=='self':
            if node.func.attr=='read_cache':
                assert len(node.args)==2 and not node.keywords and isinstance(node.args[0],ast.Name)and node.args[0].id in ('kc','vc')
                assert dump(node.args[1])==dump(expression('slots'))
                self.reads+=1
                return expression(node.args[0].id+'[slots.clamp_min(0)]')
            if node.func.attr=='write_cache':
                assert not node.keywords and len(node.args)==3
                assert [dump(x)for x in node.args]in [[dump(expression(x))for x in names]for names in [('kc','sk','slots'),('vc','sv','slots')]]
                self.writes+=1;node.func=ast.Name(id='store_slots',ctx=ast.Load())
        return node

def compare(reference,candidate):
    candidate=copy.deepcopy(candidate)
    assert len(candidate.body)>8
    early=candidate.body[0]
    assert isinstance(early,ast.If)and dump(early.test)==dump(expression('prefill'))and not early.orelse
    assert len(early.body)==1 and isinstance(early.body[0],ast.Return)
    assert dump(early.body[0].value)==dump(expression('Attention.forward(self,x,cache,slots,positions,lengths,prefill)'))
    prefix=1
    while isinstance(candidate.body[prefix],ast.Assert):prefix+=1
    assert prefix==7
    lower=LowerOnlyCacheHelpers();body=lower.visit(ast.Module(body=candidate.body[prefix:],type_ignores=[]))
    assert lower.reads==lower.writes==2
    return dump(reference.args)==dump(candidate.args)and dump(ast.Module(body=reference.body,type_ignores=[]))==dump(body)

def run():
    spec_path=Q/'smollm3-observer-sio14-execution-spec-20261108-v1.json'
    assert ref(spec_path)['sha256']=='6ae6357c45f3d6bbd0c2fc38d5a77fe24d2c537073b67ca5589554da05c14337'
    pins=json.loads(spec_path.read_text())['runtime_source_sha256'];sources={}
    for name in ('text_core.py','cache_attention.py','helpers.py','nki_write.py','nki_write_torch.py','modeling.py'):
        a,b=TARGET/name,PINNED/name
        assert ref(a)['sha256']==ref(b)['sha256']==pins[str(b)]
        sources[name]=dict(installed=ref(a),pinned_source=ref(b))
    tree=ast.parse((TARGET/'text_core.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Attention');original=function(cls,'forward')
    tree=ast.parse((TARGET/'cache_attention.py').read_text());candidate=function(tree,'attention_forward')
    read=function(tree,'read_reference');assert len(read.body)==1 and isinstance(read.body[0],ast.Return)and dump(read.body[0].value)==dump(expression('cache[slots.clamp_min(0)]'))
    write_cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NkiWriteAttention')
    assert len(write_cls.bases)==1 and dump(write_cls.bases[0])==dump(expression('NkiAttention'))
    assert len(write_cls.body)==1 and dump(write_cls.body[0])==dump(ast.parse('read_cache = staticmethod(read_reference)').body[0])
    base_cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NkiAttention')
    assert dump(base_cls.body[-1])==dump(ast.parse('write_cache = staticmethod(write_cache)').body[0])
    assert compare(original,candidate)
    controls=[]
    changed=copy.deepcopy(candidate)
    scale=next(n for n in ast.walk(changed)if isinstance(n,ast.BinOp)and isinstance(n.op,ast.Pow)and dump(n.left)==dump(expression('self.dim')))
    scale.right=expression('-0.25');assert not compare(original,changed);controls.append('changed_attention_scale_rejected')
    changed=copy.deepcopy(candidate)
    mask=next(n for n in ast.walk(changed)if isinstance(n,ast.Compare)and any(isinstance(op,ast.LtE)for op in n.ops))
    mask.ops=[ast.Lt()];assert not compare(original,changed);controls.append('changed_inclusive_mask_rejected')
    changed=copy.deepcopy(candidate)
    softmax=next(n for n in ast.walk(changed)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='softmax')
    next(k for k in softmax.keywords if k.arg=='dtype').value=expression('torch.bfloat16')
    assert not compare(original,changed);controls.append('changed_softmax_precision_rejected')
    sdk=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/spmd.py')
    old=ROOT/'smollm3_matched_state_device_v2.py'
    report=dict(schema='smollm3_sio14_attention_source_comparison_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        execution_spec=ref(spec_path),source_comparison_script=ref(Path(__file__)),sources=sources,
        reference_and_candidate_arithmetic_ast_equal_after_exact_cache_helper_substitution=True,
        candidate_prefill_delegates_original=True,input_assertions_excluded=6,read_sites_normalized=2,write_sites_normalized=2,
        write_variant_read_helper_matches_reference=True,negative_controls=controls,
        state_access_source=ref(sdk),historical_snapshot_helper_source=ref(old),
        interpretation='Python attention arithmetic is identical after substituting the two intended cache reads and two intended writes. This does not establish compiled arithmetic, NKI copy semantics, alias safety, actual cache contents, or cause of output divergence.',
        next_steps=['Finish the original12round matrix before selecting a successor.',
          'Read actual live state via traced_model.nxd_model.state only in a separately admitted observer experiment, retaining original loader/warmup and all22requests.',
          'Reuse no clear/restore/load behavior from the historical matched-state worker: it deliberately resets state and differs from the online lifecycle.',
          'Any new cache/logit observer needs its own off/on output controls, bounded complete reads, original process/source evidence, and same slot/request indexing.',
          'If histories remain different, complete native self-replay and controlled full-history comparisons before a causal kernel claim.'],
        model_or_compiler_imported=False,model_execution_performed=False,device_cache_read=False,
        state_snapshot_bytes_if_full144=603979776,full_model_qualified=False,new_speedup_claim=False)
    p=Q/'smollm3-sio14-attention-source-comparison-20261108-v1.json'
    with p.open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(report=ref(p),attention_ast_equal=True,negative_controls=controls,source_pairs_verified=len(sources))))
if __name__=='__main__':run()
