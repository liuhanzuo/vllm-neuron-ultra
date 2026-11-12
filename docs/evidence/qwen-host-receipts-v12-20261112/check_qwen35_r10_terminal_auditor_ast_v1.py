"""Source-only AST preservation check; never imports the audited program."""
import ast
from pathlib import Path
Q=Path(__file__).resolve().parent
old=ast.parse((Q/'audit_qwen35_sampler_r10_terminal_root_20261112_v1.py').read_text())
new=ast.parse((Q/'audit_qwen35_sampler_r10_terminal_root_20261112_v2.py').read_text())
dump=lambda node:ast.dump(node,include_attributes=False)
funcs=lambda tree:{node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
a,b=funcs(old),funcs(new)
unchanged=['raw','ref','read','bound','canonical_process_owner','terminal_unit','verify_current_terminal']
for name in unchanged:assert dump(a[name])==dump(b[name]),name

def anchor(body,text):
    return next(i for i,node in enumerate(body) if ast.unparse(node).startswith(text))

sa,sb=a['science'].body,b['science'].body
ia=anchor(sa,'preflight = read(')
ja=anchor(sa,"summary['native_preflight42_passed'] = True")
preflight=b['preflight_science'].body
ib=anchor(preflight,'preflight = read(')
assert list(map(dump,sa[ia:ja]))==list(map(dump,preflight[ib:-1]))
assert dump(sa[anchor(sa,'def native(')])==dump(sb[anchor(sb,'def native(')])
fail_a=anchor(sa,'if not successful:')
fail_b=anchor(sb,'if not successful:')
assert list(map(dump,sa[fail_a:]))==list(map(dump,sb[fail_b:]))
# Model gates retained; vector production now additionally requires completed producer.
loop_a=sa[anchor(sa,"for arm in ('baseline', 'candidate'):")]
loop_b=sb[anchor(sb,"for arm in ('baseline', 'candidate'):")]
original_vector_test=loop_a.body[-1].test
assert isinstance(loop_b.body[-1].test,ast.BoolOp)
assert isinstance(loop_b.body[-1].test.op,ast.And)
assert len(loop_b.body[-1].test.values)==2
assert dump(loop_b.body[-1].test.values[-1])==dump(original_vector_test)
loop_b.body[-1].test=original_vector_test
assert dump(loop_a)==dump(loop_b)
# The full phase harvesting, 20-phase/21-service success gate, lock and two drain passes are unchanged.
ma,mb=a['main'].body,b['main'].body
start_a=anchor(ma,"owners = [a['outer_owner'], co['process']]")
start_b=anchor(mb,"owners = [a['outer_owner'], co['process']]")
end_a=anchor(ma,"value = dict(schema=")
end_b=anchor(mb,"value = dict(schema=")
assert list(map(dump,ma[start_a:end_a]))==list(map(dump,mb[start_b:end_b]))
text=ast.unparse(a['main'])
assert 'len(stages) == len(PHASES) == 20' in text
assert 'len(service_ends) == 21' in text
assert 'for _ in range(2):' in text
print('PASS: unchanged original success science, extracted preflight, 20 phases, 21 services, lock and two drain passes; only completed-producer vector guard added')
