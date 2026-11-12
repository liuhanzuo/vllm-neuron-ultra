"""Future LFM full-CPU metadata binding only; no production imports or writes.

Fixed accepted592-source closure and original scientific/resource contracts.
Root supplies every new history/output/canonical input. Returned spec remains
root_finalized=False and cannot launch until root independently finalizes it.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT/'research'
REQUEST_SCHEMA = 'lfm3_full_cpu_binding_request_v1'
RESULT_SCHEMA = 'lfm3_full_cpu_binding_candidate_v1'
CANDIDATE_REF = dict(path=str(Q/'lfm3-full-cpu-execution-source-candidate-20261111-v2.json'),
    sha256='e224c5d0a5cce199d35bc9ed9dfdabab7a7f48f0407ce30254436bd24f19b4be')
ROOT_REVIEW_REF = dict(path=str(Q/'lfm3-full-cpu-v2-root-source-review-20261111-v1.json'),
    sha256='74fd9dcf6cda27a60bbb16695a3052ae5465571d5c5fbf4565488655232ea2e3')
RUNNER = str(ROOT/'run_lfm3_nki05_full_cpu_serial_v2.py')
LEAF = str(ROOT/'lfm3_nki05_full_cpu_leaf_v2.py')
TEXT_CAP = 8*2**20


def require(value,message):
    if not value:raise ValueError(message)


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def reference(ref):
    require(type(ref) is dict and set(ref) in ({'path','sha256'},{'path','sha256','bytes'})
        and type(ref['path']) is str and Path(ref['path']).is_absolute() and str(Path(ref['path']))==ref['path']
        and '..' not in Path(ref['path']).parts and type(ref['sha256']) is str
        and re.fullmatch('[a-f0-9]{64}',ref['sha256']), 'Exact explicit path/SHA reference')
    require('bytes' not in ref or type(ref['bytes']) is int and ref['bytes']>=0,'Exact optional reference length')


def same(a,b):return encoded(a)==encoded(b)


def check_pins(pins,expected):
    require(type(pins) is dict and len(pins)==592, 'All592 reviewed sources required; old570/590 maps reject')
    require(same(pins,expected), 'Every accepted592 source path and old SHA retained without rebinding or additions')


def authenticated_sources(source_ref,review_ref,read,text):
    require(source_ref==CANDIDATE_REF and review_ref==ROOT_REVIEW_REF,'Exact independently accepted full-CPU-v2 source references')
    candidate,review=read(source_ref),read(review_ref)
    require(candidate['schema']=='lfm3_full_cpu_execution_source_candidate_v2' and candidate['source_count']==592
        and candidate['execution_entrypoint']==RUNNER and candidate['runtime_admitted'] is False,
        'Original592 candidate and exact full-CPU-v2 entry')
    require(review['schema']=='lfm3_full_cpu_v2_root_source_review_v1' and review['reviewer']=='/root'
        and review['candidate']==source_ref and review['source_only_accepted'] is True
        and review['blockers_for_source']==[] and review['runtime_admitted'] is False,
        'Independent accepted source review, never execution approval')
    pins=candidate['source_sha256'];check_pins(pins,pins)
    old590=read(candidate['previous590_candidate']);old570=read(candidate['previous_spec']);old558=read(candidate['original558_closure'])
    for old,count in ((old590,590),(old570,570),(old558,558)):
        oldpins=old['source_sha256']
        require(len(oldpins)==count and all(pins.get(path)==sha for path,sha in oldpins.items()),
            'Every original558/r5-570/fullCPU-590 source preserved at the same SHA')
    raw_sources={}
    for path,sha in pins.items():
        reference(dict(path=path,sha256=sha));require(Path(path).suffix=='.py','Only accepted Python execution-source bytes')
        raw=text(path,sha)
        require(hashlib.sha256(raw).hexdigest()==sha,'Frozen execution source changed: '+path)
        raw.decode('utf-8');raw_sources[path]=raw
    require(pins[RUNNER]=='01a39c067f41b2a32e70f23e78019c9524892267fcd6f7a7d9c992279551d0aa'
        and pins[LEAF]=='cee465642e4cc7101b2e216c5cd71f727e5820c3485977c1dbed897e3fe82b16',
        'Frozen original accepted runtime and scientific leaf')
    return candidate,old570,copy.deepcopy(pins),raw_sources


def definitions(raw,path,names,namespace):
    """Only explicitly named authenticated pure definitions/constants."""
    tree=ast.parse(raw,filename=path);selected=[];found=set()
    for node in tree.body:
        assigned=set()
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):assigned={node.name}
        elif isinstance(node,ast.Assign):
            assigned={n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name)}
        if assigned & set(names):
            require(assigned<=set(names),'Whole original multi-target assignment retained')
            require(not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(node)),
                'No import in extracted metadata definition')
            selected.append(node);found|=assigned
    require(found==set(names),'All exact requested original definitions found')
    exec(compile(ast.Module(body=selected,type_ignores=[]),path,'exec',dont_inherit=True),namespace)
    return namespace


def original_contract(raw_sources):
    def load(name,names,namespace=None):
        path=str(ROOT/name);ns=dict(Path=Path,ROOT=ROOT,Q=Q,copy=copy,re=re,require=require)
        if namespace:ns.update(namespace)
        return definitions(raw_sources[path],path,names,ns)
    fixture=load('lfm3_typed_fixture_ledger_leaf_v1.py',('MATRICES',))
    protocol=load('lfm3_typed_writer_cpu_pilot_protocol_v3.py',('RUNTIME_PLAN','RUNTIME_PLAN_SHA'))
    science=load('lfm3_nki05_full_cpu_leaf_v2.py',('OLD_PYTHON','NEW_PYTHON','STAGES','PHASES',
        'BATCH_SECONDS','PHASE_SECONDS','WORKER_SECONDS','protocol_manifest'),
        dict(fixtures=NS(MATRICES=fixture['MATRICES']),legacy=NS(p=NS(**protocol))))
    manifest=science['protocol_manifest']()
    require((manifest['total_children'],manifest['full_cases'],manifest['full_batches'],manifest['full_passes'],manifest['control_children'])
        ==(874,3716,291,3,1),'Exact original scientific extent')
    history=load('lfm3_nki05_serial_history_v1.py',('MINIMUM','MINI_LIVE','retained_runs'),dict(p=NS(require=require)))
    host=load('run_lfm3_nki05_full_cpu_serial_v2.py',('FILE_CAP','OUTPUT_CAP','NVME_FLOOR','MAX_FILES','FLOOR','LOCK','PROPS','SPEC_KEYS','validate_spec'),
        dict(SELF=RUNNER,p=NS(require=require),science=NS(__file__=LEAF,protocol_manifest=science['protocol_manifest']),
             history=NS(MINIMUM=history['MINIMUM'],serial=NS(sources=NS(reference=reference)))))
    pinbase=load('historical_b3_payload_partition_v1.py',('pin_shape',))
    pinwrapper=load('nemotron9b_package_metadata_v27.py',('pin_shape',),dict(history=NS(pin_shape=pinbase['pin_shape'])))
    units=load('research/minicpm5_sdk_physical_evidence_v3.py',('unit_name',))
    sdk=load('research/minicpm5_layer1_operator_sdk_completion_v2.py',('BINDING',))
    serial=load('research/minicpm5_layer1_operator_sdk_runtime_completion_v4.py',('MAX_SERIAL_RUNS','serial_catalog'),
        dict(sdk=NS(BINDING=sdk['BINDING']),sources=NS(reference=reference,metadata=NS(pin_shape=pinwrapper['pin_shape'])),
             physical_evidence=NS(unit_name=units['unit_name'])))
    return dict(host=host,manifest=manifest,history=history,serial=serial,sdk_binding=sdk['BINDING'])


def history_metadata(catalog_ref,terminal_ref,old_spec,contract,pins,read,guard):
    """Saved metadata only; actual latest/physical/history.resolve stays runtime."""
    for ref in (catalog_ref,terminal_ref):reference(ref)
    require(catalog_ref!=contract['history']['MINIMUM'],'Original pre-Mini catalog cannot authorize this successor')
    catalog=read(catalog_ref);terminal=read(terminal_ref)
    require(catalog['current_handoff']==terminal_ref,'Explicit catalog and terminal_handoff must match exactly')
    require(terminal['live_tool_sessions']==[] and terminal['active_heavy_units']==[], 'Supplied saved terminal is idle, not live-only')
    minimum=read(contract['history']['MINIMUM']);live=read(contract['history']['MINI_LIVE'])
    contract['history']['retained_runs'](catalog,minimum,live)
    older=read(old_spec['serial_catalog']);rows={row['terminal']['path']:row for row in catalog['runs']}
    require(all(rows.get(row['terminal']['path'])==row for row in older['runs']), 'Every original r5 serial row retained verbatim')
    provenance={}
    for row in catalog['runs']:
        for path,sha in row['source_sha256'].items():
            require(path not in provenance or provenance[path]==sha,'Historical source pin conflict')
            require(path not in pins or pins[path]==sha,'History cannot rebind an accepted execution source')
            provenance[path]=sha
    # This original pure validator consumes every complete row and terminal;
    # its saved source map is provenance, never the LFM execution source map.
    later=contract['serial']['serial_catalog'](dict(serial_predecessors=catalog_ref,source_sha256=provenance),read,guard)
    return later


def prepare(request,*,read,text,guard=lambda:None,exists=os.path.lexists):
    expected={'schema','source_candidate','source_review','serial_catalog','terminal_handoff','output_root','unit','canonical_sha256'}
    require(type(request) is dict and set(request)==expected and request['schema']==REQUEST_SCHEMA,
        'Only explicit source/history/output/canonical inputs; no model/budget/science override')
    candidate,old_spec,pins,raw=authenticated_sources(request['source_candidate'],request['source_review'],read,text)
    contract=original_contract(raw);host=contract['host']
    root=Path(request['output_root']);unit=request['unit']
    require(str(root)==request['output_root'] and root.parent==Path('/mnt/nvme')
        and re.fullmatch(r'lfm3-nki05-full-cpu-[a-z0-9-]{1,80}',root.name),'Original fresh full-CPU output namespace')
    require(type(unit) is str and re.fullmatch(r'trn1-lfm3-nki05-full-cpu-[a-z0-9-]{1,80}',unit),'Original fresh full-CPU unit namespace')
    require(callable(exists) and not exists(str(root)), 'Previously existing output path cannot be reused')
    later=history_metadata(request['serial_catalog'],request['terminal_handoff'],old_spec,contract,pins,read,guard)
    require(unit not in later['units'] and unit+'.service' not in later['units'],'No reused historical unit')
    canonical=request['canonical_sha256']
    require(type(canonical) is dict and set(canonical)==set(old_spec['canonical_sha256']), 'All three original canonical metadata paths retained')
    for path,sha in canonical.items():
        require(Path(path).parent==ROOT and Path(path).suffix=='.json','Original root canonical metadata only')
        ref=dict(path=path,sha256=sha);reference(ref);read(ref)
    spec=dict(schema='lfm3_nki05_full_cpu_spec_v1',root_finalized=False,runner=RUNNER,diagnostic=LEAF,
        source_sha256=pins,serial_catalog=copy.deepcopy(request['serial_catalog']),terminal_handoff=copy.deepcopy(request['terminal_handoff']),
        output_root=str(root),unit=unit,resources=copy.deepcopy(host['PROPS']),lock_identity=copy.deepcopy(host['LOCK']),
        canonical_sha256=copy.deepcopy(canonical),root_floor_bytes=host['FLOOR'],output_cap_bytes=host['OUTPUT_CAP'],
        nvme_floor_bytes=host['NVME_FLOOR'],max_files=host['MAX_FILES'],cpu_protocol=contract['manifest'],
        model_qualification_claim=False,automatic_successor_launch=False)
    require(set(spec)==host['SPEC_KEYS'],'Exact original spec fields; root independently finalizes only after review')
    require(len(encoded(spec))<=host['FILE_CAP'],'Complete candidate fits original256KiB report/spec envelope')
    guard()
    return dict(schema=RESULT_SCHEMA,spec_candidate=spec,source_candidate=request['source_candidate'],source_review=request['source_review'],
        source_files=592,source_map_sha256=hashlib.sha256(encoded(pins)).hexdigest(),catalog_rows=len(later['runs']),
        full_children=874,full_cases_per_pass=3716,full_batches_per_pass=291,full_passes=3,control_children=1,
        exact_original_resource_and_scientific_contract=True,original592_source_map_preserved=True,
        all_original_r5_history_rows_retained=True,execution_sources_separate_from_history=True,
        spec_requires_independent_root_finalization=True,review_created=False,current_runtime_binding_written=False,
        latest_live_freshness_verified=False,physical_service_lock_checks_performed=False,
        native_RECORD_raw_imports_or_reads=False,runtime_admitted=False,launch_authorized=False,
        full_model_qualified=False,new_speedup_claim=False)


def validate_candidate(spec,request,**readers):
    """Rebuild from authenticated inputs; reject any science/budget/model drift."""
    expected=prepare(request,**readers)['spec_candidate']
    require(same(spec,expected),'Candidate changed: only independent root finalization is a subsequent root decision')
    return True


class Reader:
    """Ordinary bounded text reader; no RECORD/native/tensor file class."""
    def __init__(self):self.identities={}
    @staticmethod
    def identity(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    def text(self,path,sha):
        reference(dict(path=path,sha256=sha));p=Path(path)
        require(p.suffix in ('.json','.py') and p.resolve()==p and not any(x.is_symlink() for x in (p,*p.parents)),
            'Only canonical Python/JSON text; no native/RECORD/raw reads')
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and before.st_size<=TEXT_CAP,'Original ordinary8MiB text cap')
            with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(TEXT_CAP+1)
            after=os.fstat(fd);now=p.stat();ident=self.identity(before)
            require(len(raw)==before.st_size and len(raw)<=TEXT_CAP and ident==self.identity(after)==self.identity(now), 'Whole stable bounded text')
            require(hashlib.sha256(raw).hexdigest()==sha,'Exact text hash: '+path);raw.decode('utf-8')
            require(path not in self.identities or self.identities[path]==ident,'No source/metadata identity rebound')
            self.identities[path]=ident;return raw
        finally:os.close(fd)
    def read(self,ref):
        reference(ref);require(Path(ref['path']).suffix=='.json','Metadata JSON only')
        raw=self.text(ref['path'],ref['sha256']);require('bytes' not in ref or len(raw)==ref['bytes'],'Exact metadata length')
        return json.loads(raw)
    def unchanged(self):
        require(all(self.identity(Path(path).stat())==ident for path,ident in self.identities.items()),'All original metadata/source identities unchanged')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args();reader=Reader()
    request=reader.read(dict(path=args.request,sha256=args.request_sha256))
    result=prepare(request,read=reader.read,text=reader.text,exists=os.path.lexists)
    reader.unchanged();print(json.dumps(result,sort_keys=True,indent=2),flush=True)


if __name__=='__main__':main()
