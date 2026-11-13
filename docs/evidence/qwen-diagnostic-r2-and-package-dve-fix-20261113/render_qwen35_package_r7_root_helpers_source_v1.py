"""Pure r7 source renderer: exact observed PWP/DVE assets, explicit root bindings.

No filesystem/native IO, History, lock or current-state selection on import or
render. Root must supply reviewed predecessor bytes and future current refs.
Returned payloads remain unadmitted source candidates until root checks them.
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
    'prepare': ('prepare_qwen35_package_full_root_r6_20261112_v1.py', 'aa6fc78e26f746bb0a0d285c28d705fbf56aa7f10d18d3a4fff1fc6d1e3eca04'),
    'accept': ('accept_qwen35_package_full_root_r6_20261112_v1.py', '28cdb748384576c3da3280ff38bc0de15b69bcf1587278c933a20953aec4f95e'),
    'ack': ('ack_qwen35_package_full_r6_root_20261112_v1.py', '2f4d6307033adc41da1375e92130691e5ffbc82c24bc903fea6778d7b0d90239'),
    'observer': ('/home/ubuntu/qwen35_package_full_r6_post_exit_20261112_v1.py', 'ed5099721682e40b7928838100341236023213cb42b8a473b744e6ba72da5a25'),
    'budget': ('qwen35-package-full-r6-20261112-resource-budget.json', '0910b5af4820b455e702f8fffc60e96d85c0f92df6e96d10f5eb4d497a82f459'),
    'stat-inputs': ('qwen35-package-full-r6-20261112-stat-inputs.json', '86f5abb733e1427bbb9f086b5cc5fd4b840c2296766151588e412cedc3d9b188'),
}
AUDIT_PINS = {
    'post_audit': (str(Q/'qwen35_package_post_audit_v2.py'), '041eeb28e30b8248d98cae42981273316f98e008746ba456e10c0045911f15b2'),
    'physical': (str(Q/'qwen35_package_audit_physical_v2.py'), '5fffef1f7d89bc0d79c60ac2c14d8ce61b4c0c2dd484a752e30f84ac4a0e9490'),
}
RUNTIME_DELIVERY = dict(path=str(Q/'qwen35_package_serial_runtime_v9/source_delivery.json'), sha256='7fc6b85ff1f241dbc548e85d6bbc3ef3a208b10d5592ad4559307cb3a8a5e12a')
PREPARER_DELIVERY = dict(path=str(Q/'qwen35-package-serial-preparer-v8-source-delivery-20261113-v1.json'), sha256='5963fed6d71e5db8220b0356ed7471fd3d8a6e84c2337090c7527f4dc81ae145')


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
    require(type(text) is str and re.fullmatch('[0-9]{8}',text),'canonical YYYYMMDD date required')
    value=date(int(text[:4]),int(text[4:6]),int(text[6:]))
    require(value>=date(2026,11,13),'post-r6 binding date required')
    return value


def successor_name(text):
    """Only the r6-owned namespace moves; seed evidence retains its date/name."""
    text=text.replace('r6','r7')
    for before,after in (
        ('full_root_r7_20261112','full_root_r7_20261113'),
        ('full_r7_root_20261112','full_r7_root_20261113'),
        ('full_r7_post_exit_20261112','full_r7_post_exit_20261113'),
        ('full-r7-20261112','full-r7-20261113'),
        ('full-r7-root-binding-20261112','full-r7-root-binding-20261113'),
        ('full-r7-root-live-20261112','full-r7-root-live-20261113'),
        ('pkg-r7-20261112','pkg-r7-20261113'),
        ('-20261112-r7','-20261113-r7')):
        text=text.replace(before,after)
    return text


def successor_metadata(budget, inputs):
    future_budget=copy.deepcopy(budget);future_inputs=copy.deepcopy(inputs)
    for key in ('model_path','artifact','compiler_cache','compiler_workdir'):
        require(budget['case'][key].endswith('-20261112-r6'),'exact predecessor r6 case path')
        future_budget['case'][key]=successor_name(budget['case'][key])
    for key in ('output_root','control_root'):
        require('pkg-r6-20261112' in budget[key],'exact prior r6 output/control')
        future_budget[key]=successor_name(budget[key])
    future_inputs['case']=copy.deepcopy(future_budget['case'])
    future_inputs['output_root']=future_budget['output_root'];future_inputs['run_name']='pkg-r7-20261113'
    raw=encode(future_budget)
    future_inputs['resource_budget']=dict(path=str(Q/'qwen35-package-full-r7-20261113-resource-budget.json'),sha256=hashlib.sha256(raw).hexdigest())
    return raw,encode(future_inputs)


def render(predecessors, bindings, audit_sources):
    require(type(predecessors) is dict and set(predecessors)==set(PINS),'all exact frozen predecessor payloads')
    for key,(_,sha) in PINS.items():
        require(type(predecessors[key]) is bytes and len(predecessors[key])<4*2**20 and
                hashlib.sha256(predecessors[key]).hexdigest()==sha,'frozen predecessor changed: '+key)
    require(type(audit_sources) is dict and set(audit_sources)==set(AUDIT_PINS),'complete exact static audit sources')
    for key,(_,sha) in AUDIT_PINS.items():
        require(type(audit_sources[key]) is bytes and len(audit_sources[key])<4*2**20 and
                hashlib.sha256(audit_sources[key]).hexdigest()==sha,'frozen audit support changed: '+key)
        ast.parse(audit_sources[key])
    require(type(bindings) is dict and set(bindings)=={'catalog','idle','running_handoff','catalog_rows','runtime_review','preparer_review'},
            'explicit actual root binding and independent review slots required')
    for key in ('catalog','idle','runtime_review','preparer_review'):reference(bindings[key])
    match=re.fullmatch(r'minicpm5-operator-post-r2-serial-workloads-root-catalog-([0-9]{8})-v([0-9]+)\.json',Path(bindings['catalog']['path']).name)
    require(match and int(match[2])>=119,'complete post-r6 catalog required')
    idle=re.fullmatch(r'continuation-handoff-([0-9]{8})-v([0-9]+)-[a-z0-9-]+\.json',Path(bindings['idle']['path']).name)
    running=Path(bindings['running_handoff'])
    run=re.fullmatch(r'continuation-handoff-([0-9]{8})-v([0-9]+)-qwen35-package-full-r7-running\.json',running.name)
    require(idle and int(idle[2])>=598 and run and int(run[2])>int(idle[2]) and
            running.parent==Q and str(running)==bindings['running_handoff'],'explicit fresh r7 running handoff required')
    dates=[calendar_date(v[1]) for v in (match,idle,run)]
    require(dates[2]>=max(dates[:2]),'running handoff date cannot precede catalog or idle')
    rows=bindings['catalog_rows'];require(type(rows) is int and rows>=166,'complete post-r6 catalog row count required')
    for key,stem in (('runtime_review','qwen35-package-serial-runtime-v9-root-source-review'),
                     ('preparer_review','qwen35-package-serial-preparer-v8-root-source-review')):
        name=Path(bindings[key]['path']).name
        m=re.fullmatch(stem+r'-([0-9]{8})-v([1-9][0-9]*)\.json',name)
        require(m is not None,'explicit independent root review for exact source version required')
        calendar_date(m[1])
    raw_budget,raw_inputs=successor_metadata(json.loads(predecessors['budget']),json.loads(predecessors['stat-inputs']))
    def target(key):
        name=Path(PINS[key][0]);return successor_name(str(name if name.is_absolute() else Q/name))
    files={target('budget'):raw_budget,target('stat-inputs'):raw_inputs}
    refs={key:copy.deepcopy(bindings[key]) for key in ('catalog','idle')}
    refs.update({'runtime-review':copy.deepcopy(bindings['runtime_review']),'preparer-review':copy.deepcopy(bindings['preparer_review'])})
    for key in ('budget','stat-inputs'):refs[key]=dict(path=target(key),sha256=hashlib.sha256(files[target(key)]).hexdigest())
    refs['candidate']=dict(path=str(Q/'qwen35-package-full-r7-20261113-binding-candidate.json'),sha256=None)
    audit_pins={path:sha for path,sha in AUDIT_PINS.values()};logs={}
    for key in ('accept','ack','observer','prepare'):
        original=predecessors[key].decode();text=original;changes=[]
        def change(before,after):
            nonlocal text
            count=text.count(before)
            require(count>0 and before!=after,'required narrow source edit absent: '+key)
            text=text.replace(before,after);changes.append(dict(before=before,after=after,count=count))
        def assignment(name,value):
            nodes=[n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
            require(len(nodes)==1,'exact single assignment: '+name)
            change(ast.get_source_segment(text,nodes[0]),name+' = '+repr(value))
        # Each owned namespace substitution is independently reversible.
        change('r6','r7')
        for before,after in (
            ('full_root_r7_20261112','full_root_r7_20261113'),
            ('full_r7_root_20261112','full_r7_root_20261113'),
            ('full_r7_post_exit_20261112','full_r7_post_exit_20261113'),
            ('full-r7-20261112','full-r7-20261113'),
            ('full-r7-root-binding-20261112','full-r7-root-binding-20261113'),
            ('full-r7-root-live-20261112','full-r7-root-live-20261113'),
            ('pkg-r7-20261112','pkg-r7-20261113'),
            ('-20261112-r7','-20261113-r7')):
            if before in text:change(before,after)
        for before,after in (
            ('qwen35_package_serial_runtime_v8','qwen35_package_serial_runtime_v9'),
            ('prepare_qwen35_package_serial_candidate_v7','prepare_qwen35_package_serial_candidate_v8'),
            ('qwen35_package_serial_preparer_v7','qwen35_package_serial_preparer_v8'),
            ('qwen35_package_post_audit_v1','qwen35_package_post_audit_v2'),
            ('qwen35_package_audit_physical_v1','qwen35_package_audit_physical_v2'),
            ('runtime v8 and preparer v7','runtime v9 and preparer v8'),
            ('qwen35-package-serial-preparer-v7-source-delivery-20261112.json',Path(PREPARER_DELIVERY['path']).name),
            ('92023b408ab8966518aa502db6a44b1b4937a1c288f1cb00c869a17087fa1986',PREPARER_DELIVERY['sha256']),
            ('8ef03038fd82bb3b3f0fe8c07a5dcad5ed7e24571dca434bd1b482a13257ecc7',RUNTIME_DELIVERY['sha256']),
            ('continuation-handoff-20261113-v597-qwen35-package-full-r7-running.json',running.name)):
            if before in text:change(before,after)
        if key in ('prepare','accept'):
            node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXACT_REFS' for t in n.targets))
            names=ast.literal_eval(node.value)
            assignment('EXACT_REFS',{name:(refs[name]['path'],refs[name]['sha256']) for name in names})
        if key=='observer':
            change("require(len(old['runs']) == 165, 'exact retained r7 predecessor has 165 rows')",
                   f"require(len(old['runs']) == {rows}, 'exact retained r7 predecessor has {rows} rows')")
            assignment('audit_support_pins',audit_pins)
        if key=='prepare':
            helper_pins=dict(audit_pins)
            for helper in ('ack','observer'):helper_pins[target(helper)]=hashlib.sha256(files[target(helper)]).hexdigest()
            assignment('expected_audit_helper_pins',helper_pins)
        restored=text
        for row in reversed(changes):
            require(restored.count(row['after'])==row['count'],'source reversal must be unambiguous')
            restored=restored.replace(row['after'],row['before'])
        require(restored==original,'entire exact r6 predecessor recoverable')
        ast.parse(text);files[target(key)]=text.encode();logs[target(key)]=changes
    require(len(files)==6,'all six source/metadata outputs required')
    return dict(schema='qwen35_package_r7_rendered_source_candidate_v1',source_only=True,passed=False,launch_authorized=False,
        actual_bindings_checked=False,actual_prepare_executed=False,history_resolved=False,lock_acquired=False,native_executed=False,
        files=files,exact_predecessor_reversal=logs,required_actual_refs=refs,inherited_stat_identities_pending_actual_recheck=True,
        audit_source_prebinding=dict(static_sources=audit_pins,generated_and_static_helpers=helper_pins,
            all_helpers_in_preparation_sources=True,all_helpers_required_before_prepare=True),
        all25_scientific_stages_unchanged=True,original_accept_resource_limits_unchanged=True,
        original_full_history_and_874_children_unchanged=True,post_exit_request_schema='qwen35_package_r7_post_exit_request_v1')
