"""Pure r6 source renderer v2: dated serial bindings; no IO or executable CLI.

Caller supplies authenticated predecessor bytes and explicit current root
bindings only after real catalog/idle publication. Returned bytes are review
material, not published files, current state, runtime admission or execution.
"""
import ast
import copy
from datetime import date
import hashlib
import json
from pathlib import PurePosixPath as Path
import re

Q = Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
PINS = {
    'prepare': ('prepare_qwen35_package_full_root_r5_20261112_v1.py', '14cb09c2009f4ff63a96ad1dde78dbd8443d164bc768ea9358a4dcba19425e19'),
    'accept': ('accept_qwen35_package_full_root_r5_20261112_v1.py', 'c199052c4d9a264f5295d2abe6cb93b82f80dcec3f9016f604d0f3a314ebeb56'),
    'ack': ('ack_qwen35_package_full_r5_root_20261112_v1.py', '33603f47eabb7fbe950dbdcd0c7c085af867c827b512d929bf0dec73c7344332'),
    'observer': ('/home/ubuntu/qwen35_package_full_r5_post_exit_20261112_v1.py', 'dc78d2eddcf1f805dd3b23c5a8b4127a4089caaeaf67fdfb863ffe6c6ccfb29e'),
    'budget': ('qwen35-package-full-r5-20261112-resource-budget.json', '904ef9e38a32758896f5c5d8016d4ab7270327c1eb531ce0397516f2ab9b3510'),
    'stat-inputs': ('qwen35-package-full-r5-20261112-stat-inputs.json', 'f4d77cfdc00c754bb9639e7720e21e1f802222dca25a7d5c5a0fc32b24e43d6b'),
}
RUNTIME_REVIEW = dict(path=str(Q/'qwen35-package-serial-runtime-v8-root-source-review-20261112-v1.json'),
    sha256='be7407d2f73e4011779ac8bbd4820e867edc9baf94c2f0dc3d07730c516f299a')
PREPARER_REVIEW = dict(path=str(Q/'qwen35-package-serial-preparer-v7-root-source-review-20261112-v1.json'),
    sha256='44df166b46fcd6173649a5231a09429b33a8b895689e93a4c5b53e1755a26ac7')


def require(value, message):
    if not value: raise ValueError(message)


def encode(value):
    payload=(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    require(len(payload)<=8*2**20,'unchanged 8 MiB metadata cap')
    return payload


def reference(value):
    require(type(value) is dict and set(value)=={'path','sha256'},'exact explicit reference')
    path=Path(value['path'])
    require(path.is_absolute() and str(path)==value['path'] and '..' not in path.parts and
            path.parent==Q and path.suffix=='.json' and re.fullmatch('[a-f0-9]{64}',value['sha256']),
            'canonical root metadata path and SHA required')


def calendar_date(text):
    """Only canonical YYYYMMDD calendar dates from the original run onward."""
    require(type(text) is str and re.fullmatch('[0-9]{8}',text),'canonical YYYYMMDD date required')
    value=date(int(text[:4]),int(text[4:6]),int(text[6:]))
    require(value>=date(2026,11,12),'serial binding date predates original r6 namespace')
    return value


def successor_metadata(budget, inputs):
    """Named future paths only. All inherited identities await actual recheck."""
    future_budget=copy.deepcopy(budget); future_inputs=copy.deepcopy(inputs)
    for key in ('model_path','artifact','compiler_cache','compiler_workdir'):
        before=budget['case'][key]
        require(before.endswith('-20261112-r5'),'exact prior r5 case path')
        future_budget['case'][key]=before[:-2]+'r6'
    for key in ('output_root','control_root'):
        require('pkg-r5-20261112' in budget[key],'exact prior r5 output/control')
        future_budget[key]=budget[key].replace('pkg-r5-20261112','pkg-r6-20261112')
    future_inputs['case']=copy.deepcopy(future_budget['case'])
    future_inputs['output_root']=future_budget['output_root'];future_inputs['run_name']='pkg-r6-20261112'
    raw_budget=encode(future_budget)
    future_inputs['resource_budget']=dict(path=str(Q/'qwen35-package-full-r6-20261112-resource-budget.json'),
                                         sha256=hashlib.sha256(raw_budget).hexdigest())
    return raw_budget,encode(future_inputs)


def render(predecessors, bindings):
    """Return source/metadata bytes; never validates existence or publishes.

    bindings must come from root's authenticated current catalog/idle read.
    Synthetic bindings are allowed only in tests and must never be published.
    """
    require(type(predecessors) is dict and set(predecessors)==set(PINS),'all exact frozen predecessor payloads')
    for key,(_,sha) in PINS.items():
        require(type(predecessors[key]) is bytes and len(predecessors[key])<4*2**20 and
                hashlib.sha256(predecessors[key]).hexdigest()==sha,'frozen predecessor changed: '+key)
    require(type(bindings) is dict and set(bindings)=={'catalog','idle','running_handoff','catalog_rows'},
            'explicit actual root binding slots required')
    for key in ('catalog','idle'):reference(bindings[key])
    match=re.fullmatch(r'minicpm5-operator-post-r2-serial-workloads-root-catalog-([0-9]{8})-v([0-9]+)\.json',Path(bindings['catalog']['path']).name)
    require(match and int(match[2])>=117,'post-r5 complete catalog required')
    idle_match=re.fullmatch(r'continuation-handoff-([0-9]{8})-v([0-9]+)-[a-z0-9-]+\.json',Path(bindings['idle']['path']).name)
    running=Path(bindings['running_handoff'])
    run_match=re.fullmatch(r'continuation-handoff-([0-9]{8})-v([0-9]+)-qwen35-package-full-r6-running\.json',running.name)
    require(idle_match and int(idle_match[2])>=594 and run_match and int(run_match[2])>int(idle_match[2]) and
            running.parent==Q and str(running)==bindings['running_handoff'],'explicit fresh r6 running handoff required')
    catalog_date,idle_date,running_date=(calendar_date(value[1]) for value in (match,idle_match,run_match))
    require(running_date>=max(catalog_date,idle_date),'running handoff date cannot precede selected catalog or idle')
    rows=bindings['catalog_rows']
    require(type(rows) is int and rows>=164,'complete actual post-geometry catalog row count required')
    budget,inputs=successor_metadata(json.loads(predecessors['budget']),json.loads(predecessors['stat-inputs']))
    files={str(Q/'qwen35-package-full-r6-20261112-resource-budget.json'):budget,
           str(Q/'qwen35-package-full-r6-20261112-stat-inputs.json'):inputs}
    refs={key:copy.deepcopy(bindings[key]) for key in ('catalog','idle')}
    refs.update({'runtime-review':copy.deepcopy(RUNTIME_REVIEW),'preparer-review':copy.deepcopy(PREPARER_REVIEW)})
    for key,payload in (('budget',budget),('stat-inputs',inputs)):
        path=str(Q/('qwen35-package-full-r6-20261112-'+('resource-budget' if key=='budget' else key)+'.json'))
        refs[key]=dict(path=path,sha256=hashlib.sha256(payload).hexdigest())
    refs['candidate']=dict(path=str(Q/'qwen35-package-full-r6-20261112-binding-candidate.json'),sha256=None)
    logs={}
    substitutions=[('qwen35_package_serial_runtime_v7','qwen35_package_serial_runtime_v8'),
        ('qwen35-package-serial-runtime-v7','qwen35-package-serial-runtime-v8'),
        ('prepare_qwen35_package_serial_candidate_v6','prepare_qwen35_package_serial_candidate_v7'),
        ('qwen35_package_serial_preparer_v6','qwen35_package_serial_preparer_v7'),
        ('qwen35-package-serial-preparer-v6','qwen35-package-serial-preparer-v7'),
        ('4fc890f9286787dd60c5c74be519e94adff9b9c9074777dd20092d99e04785ea','92023b408ab8966518aa502db6a44b1b4937a1c288f1cb00c869a17087fa1986'),
        ('29ecb016b48cf3e6431a62c81510cd5877bc14816b9863fa0805da7b429997ee','8ef03038fd82bb3b3f0fe8c07a5dcad5ed7e24571dca434bd1b482a13257ecc7'),
        ('runtime v7 and preparer v6','runtime v8 and preparer v7'),('r5','r6'),
        ('continuation-handoff-20261112-v590-qwen35-package-full-r6-running.json',running.name)]
    for key in ('prepare','accept','ack','observer'):
        original=predecessors[key].decode();text=original;changes=[]
        def change(before,after):
            nonlocal text
            count=text.count(before)
            if count:
                text=text.replace(before,after);changes.append(dict(before=before,after=after,count=count))
        for before,after in substitutions:change(before,after)
        if key in ('prepare','accept'):
            node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and
                any(isinstance(t,ast.Name) and t.id=='EXACT_REFS' for t in n.targets))
            old=ast.literal_eval(node.value)
            selected={name:(refs[name]['path'],refs[name]['sha256']) for name in old}
            change(ast.get_source_segment(text,node),'EXACT_REFS = '+repr(selected))
        if key=='observer':
            change("require(len(old['runs']) == 162, 'exact retained r6 predecessor has 162 rows')",
                f"require(len(old['runs']) == {rows}, 'exact retained r6 predecessor has {rows} rows')")
        restored=text
        for row in reversed(changes):
            require(restored.count(row['after'])==row['count'],'source replacement reversal ambiguous')
            restored=restored.replace(row['after'],row['before'])
        require(restored==original,'entire predecessor source must be recoverable exactly')
        ast.parse(text)
        old_path=Path(PINS[key][0]);target=old_path if old_path.is_absolute() else Q/old_path
        target=Path(str(target).replace('r5','r6'))
        files[str(target)]=text.encode();logs[str(target)]=changes
    return dict(schema='qwen35_package_r6_rendered_source_candidate_v1',source_only=True,
        passed=False,launch_authorized=False,actual_bindings_checked=False,actual_prepare_executed=False,
        history_resolved=False,lock_acquired=False,native_executed=False,files=files,
        exact_predecessor_reversal=logs,required_actual_refs=refs,
        inherited_stat_identities_pending_actual_recheck=True,
        post_exit_request_schema='qwen35_package_r6_post_exit_request_v1')
