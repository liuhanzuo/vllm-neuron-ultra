"""Pure future argv construction; no read, write, native, or process operation.

Inputs are exact reviewed source bytes and a supplied role-specific plan. The
caller remains responsible for the original launch/ACK/owner/release protocol,
source/native authority, budgets, original lock and external actual argv audit.
"""
import hashlib
import json

OUTER = '/home/ubuntu/venvs/model_reference_2026/bin/python'
SDK = '/home/ubuntu/venvs/vllm_neuron/bin/python'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode()


def reference(ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'}
        and type(ref['path']) is str and ref['path'].startswith('/')
        and all(p and p not in ('.','..') for p in ref['path'].split('/')[1:])
        and type(ref['sha256']) is str and len(ref['sha256'])==64
        and all(c in '0123456789abcdef' for c in ref['sha256']), 'Exact reviewed source/plan reference')


def launch(entry_bytes,entry_ref,plan,plan_ref,*,outer_deadline=None):
    """Return argv only; never execute and never construct an accepted review."""
    reference(entry_ref);reference(plan_ref)
    require(type(entry_bytes) is bytes and 0<len(entry_bytes)<65536
        and hashlib.sha256(entry_bytes).hexdigest()==entry_ref['sha256'],
        'Whole reviewed entry source, not a path to an unchecked script')
    require(hashlib.sha256(encoded(plan)).hexdigest()==plan_ref['sha256'],
        'Exact canonical future entry plan bytes')
    require(type(plan) is dict and plan.get('schema')=='minicpm5_source_entry_plan_v1'
        and plan.get('role') in ('outer312','sdk310')
        and plan.get('entry_sha256')==entry_ref['sha256'], 'Exact per-role future plan')
    role=plan['role']
    require(plan['python']==(OUTER if role=='outer312' else SDK), 'Original per-role interpreter')
    require(plan['native_authority']['role']==role
        and plan['native_authority']['python']==plan['python'], 'Authority belongs to this interpreter role')
    require(type(plan['successor_input']) is str and 0<len(plan['successor_input'])<=4096
        and '\0' not in plan['successor_input'], 'Explicit unchanged successor input, not an embedded command')
    if role=='outer312':
        require(outer_deadline is None, 'Outer entry establishes its only deadline')
        deadline='new'
    else:
        require(type(outer_deadline) in (int,float) and 0<outer_deadline<float('inf'),
                'SDK child must receive existing actual outer monotonic deadline')
        deadline=repr(outer_deadline)
    argv=[plan['python'],'-I','-S','-B','-X','pycache_prefix=/dev/null','-c',entry_bytes.decode('utf-8'),
        '--mini-source-entry',role,plan_ref['path'],plan_ref['sha256'],plan['successor_input'],entry_ref['sha256'],deadline]
    require(all(type(x) is str and '\0' not in x and len(x.encode())<65536 for x in argv),
            'Bounded NUL-free actual exec argv; no shell interpolation')
    return argv


def separate_authorities(outer,worker):
    """Prevent accidental reuse; native authenticity is still independently checked."""
    require(outer['role']=='outer312' and worker['role']=='sdk310'
        and outer['python']==OUTER and worker['python']==SDK,
        'Both exact original interpreter roles')
    a,b=outer['native_authority'],worker['native_authority']
    for row,plan in ((a,outer),(b,worker)):
        require(row['role']==plan['role'] and row['python']==plan['python'], 'Separate authority role')
        reference({k:row[k] for k in ('path','sha256')})
    require(a['path']!=b['path'] and a['sha256']!=b['sha256']
        and outer['executable']!=worker['executable'], 'SDK310 cannot reuse outer312 native evidence')
    return True
