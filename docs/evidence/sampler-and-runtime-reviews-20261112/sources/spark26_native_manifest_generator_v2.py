"""Populate the exact frozen provider manifest using only source/text metadata.

Never imports installed packages, runs ELF tools, opens binaries or weights,
changes packages, or creates a runtime admission. Unknown dependencies remain
explicit blockers; a schema-valid candidate is not a proven runtime closure.
"""
import argparse
import ast
import base64
import binascii
import csv
from datetime import datetime,timezone
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from urllib.parse import unquote,urlparse

import spark26_manifest_requirements_v2 as req
import spark26_native_provider_v2 as provider

ROOT,Q,SITE=provider.contract.ROOT,provider.contract.Q,provider.SITE
ENV=dict(implementation_name='cpython',implementation_version='3.10.12',os_name='posix',
    platform_machine='x86_64',platform_python_implementation='CPython',platform_system='Linux',
    python_full_version='3.10.12',python_version='3.10',sys_platform='linux')
ROOT_EXTRAS={'safetensors':['torch'],'transformers':['torch']}
REVIEWS=(
 ('qwen35-sampler-r6-20261112-review.json','f78e8882589f183940b58097e6011fdb29cfb86044879c9950cd61f1372e5cdc','passed'),
 ('spark26-neuron-port-root-source-review-20261112-v1.json','f9accae238075c5ddd7349d2a9cef4bb51a9705778af0dfa3f975d704a807260','source_only_accepted'),
 ('spark26-neuron-port-root-source-review-20261112-v2.json','7d46b9515518d21ad106e776e6edccbfce9892642efbce61a4772dc4d937db56','source_only_accepted'),
 ('spark26-native-port-diagnostic-root-source-review-20261112-v1.json','2257c622cbbb483ae45abe6f5cd51acfb0f728d1cda08e3c7cb434db3540d509','source_scope_accepted'),
 ('spark26-native-port-diagnostic-root-source-review-20261112-v2.json','79730e4eee5d7d49f8a5e78d26b5bfc4f68e09c66123e39e14cedd6c93d9dc52','source_scope_accepted'),
 ('spark26-native-provider-worker-root-source-review-20261112-v1.json','c5370b8be386d271554328d358f683c0f0fa56077f52c8eac4c1636ecb947421','source_component_review_passed'),
 ('spark26-native-serial-outer-root-source-review-20261112-v1.json','44daee3ac3c31c3029598870402ffe54850f46088ae4f4435ad8bbd700679b77','source_component_review_passed'),
)
ELF_EVIDENCE=(
 {'path':'/mnt/nvme/dory-serial-inventory-r5-20261111/inventory/external-closure.json',
  'sha256':'8e97c912e0cbcd75ee4eead45565a05d727c5bc043ef5bf0ea2eeb51002ffd6e'},
 {'path':'/mnt/nvme/dory-loader-candidate-elf-r3-20261112/result.json',
  'sha256':'0f3b1ad66402d8dd9c22e64072af9d035285036e19ba6ac5362eced694f2fb1a'},
)
TEXT_NAMES={'RECORD','METADATA','WHEEL','INSTALLER','REQUESTED','LICENSE','COPYING','AUTHORS'}
require=provider.contract.require


def encoded(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def reference(path,raw):return dict(path=str(path),sha256=digest(raw))


class TextReader:
    def __init__(self):self.references={};self.bytes=0
    def raw(self,path,wanted=None):
        path=Path(path)
        require(path.is_absolute() and path.resolve()==path and not any(p.is_symlink()for p in (path,*path.parents)),
                'Canonical source/metadata only')
        require('.so'not in path.name and (path.name in TEXT_NAMES or path.suffix in ('.py','.json','.txt','.md','.cfg','.pth')),
                'No native, archive, bytecode, weight or raw payload read')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=64*2**20,
                                         'Bounded singleton source/metadata')
            with os.fdopen(os.dup(fd),'rb')as stream:raw=stream.read(64*2**20+1)
            after=os.fstat(fd)
            require(len(raw)==before.st_size and all(getattr(before,k)==getattr(after,k)for k in
                ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Stable complete source/metadata')
        finally:os.close(fd)
        raw.decode('utf8')
        require(wanted is None or digest(raw)==wanted,'Exact frozen evidence digest')
        ref=reference(path,raw)
        require(str(path)not in self.references or self.references[str(path)]==ref,'Text input changed during preparation')
        self.references[str(path)]=ref;self.bytes+=len(raw)
        return raw
    def load(self,ref):return json.loads(self.raw(ref['path'],ref['sha256']))


def installed_metadata(reader):
    inventory={}
    for path in sorted(SITE.glob('*.dist-info/METADATA')):
        raw=reader.raw(path);metadata=Parser().parsestr(raw.decode())
        name=req.name(metadata['Name']);version=metadata['Version']
        require(name not in inventory,'Ambiguous installed distribution metadata: '+name)
        suffix='-'+version+'.dist-info'
        require(path.parent.name.endswith(suffix),'METADATA version matches exact dist-info directory')
        wheel_name=path.parent.name[:-len(suffix)]
        require(req.name(wheel_name)==name,'Normalized METADATA and dist-info identity agree')
        inventory[name]=dict(name=name,wheel_name=wheel_name,version=version,directory=str(path.parent),
            metadata=reference(path,raw),requires=metadata.get_all('Requires-Dist',[]),
            extras=sorted({req.name(e)for e in metadata.get_all('Provides-Extra',[])}),
            requires_python=metadata.get('Requires-Python',''))
    require(len(inventory)<=1024,'Finite installed metadata inventory')
    return inventory


def dependency_graph(inventory,roots,environment,root_extras):
    selected={name:set(root_extras.get(name,()))for name in roots};queue=list(roots);processed={}
    unresolved=[];edges={}
    while queue:
        name=queue.pop(0);extras=selected[name]
        if processed.get(name)==extras:continue
        processed[name]=set(extras)
        if name not in inventory:
            unresolved.append(dict(kind='missing_installed_distribution',name=name));continue
        row=inventory[name]
        try:req.version(row['version'])
        except req.Unresolved as error:unresolved.append(dict(kind='unsupported_installed_version',name=name,error=str(error)))
        if name in roots and row['version']!=roots[name]:
            unresolved.append(dict(kind='original_root_version_changed',name=name,expected=roots[name],actual=row['version']))
        unknown=extras-set(row['extras'])
        if unknown:unresolved.append(dict(kind='undeclared_requested_extras',name=name,extras=sorted(unknown)))
        try:
            if not req.satisfies(environment['python_full_version'],row['requires_python']):
                unresolved.append(dict(kind='requires_python_mismatch',name=name,requirement=row['requires_python']))
        except req.Unresolved as error:unresolved.append(dict(kind='unknown_requires_python',name=name,error=str(error)))
        for index,text in enumerate(row['requires']):
            edge=dict(parent=name,requirement=text,parent_extras=sorted(extras))
            try:
                child=req.requirement(text);edge.update(child)
                active=req.marker(child['marker'],environment,extras);edge['active']=active
                if active is False:edges[(name,index)]=edge;continue
                if active is None:unresolved.append(dict(kind='unknown_marker_environment',parent=name,requirement=text))
                dep=child['name']
                if dep not in selected:selected[dep]=set(child['extras']);queue.append(dep)
                elif not set(child['extras'])<=selected[dep]:selected[dep].update(child['extras']);queue.append(dep)
                if child['url']:
                    unresolved.append(dict(kind='direct_requirement_URL_not_resolved',parent=name,requirement=text))
                elif dep in inventory:
                    satisfied=req.satisfies(inventory[dep]['version'],child['specifiers']);edge['installed_version_satisfies']=satisfied
                    edge['installed_version']=inventory[dep]['version']
                    if not satisfied:unresolved.append(dict(kind='installed_version_mismatch',parent=name,requirement=text,actual=inventory[dep]['version']))
            except req.Unresolved as error:
                edge['active']=None;edge['unresolved']=str(error)
                unresolved.append(dict(kind='unsupported_requirement_or_marker',parent=name,requirement=text,error=str(error)))
            edges[(name,index)]=edge
        require(len(selected)<=1024 and len(edges)<=100000,'Finite dependency fixed point')
    unique={provider.contract.digest(row):row for row in unresolved}
    return dict(selected={n:sorted(v)for n,v in sorted(selected.items())},edges=list(edges.values()),unresolved=list(unique.values()),
                complete_declared_dependency_resolution=not unique,
                optional_extras_policy='Base plus explicitly requested/propagated extras; inactive conditions retained',
                not_a_complete_dynamic_import_or_ELF_closure=True)


def record_entries(text):
    """Independent reconstruction of frozen Runtime.record_payloads semantics."""
    result={}
    for relative,hash_text,size in csv.reader(text.splitlines()):
        relative_path=Path(relative)
        if relative_path.is_absolute() or '..'in relative_path.parts:continue
        if relative_path.suffix=='.pyc' or '__pycache__'in relative_path.parts:continue
        if not hash_text:
            require(relative_path.name=='RECORD','Unknown unhashed package payload');continue
        require(hash_text.startswith('sha256=') and size.isdigit(),'SHA256 RECORD row')
        token=hash_text[7:];sha=base64.urlsafe_b64decode(token+'='*(-len(token)%4)).hex()
        require(len(sha)==64,'Complete RECORD SHA256')
        path=str(SITE/relative)
        require(path not in result,'Duplicate RECORD payload pathname')
        result[path]=dict(sha256=sha,bytes=int(size))
    return result


def reviewed_sources(reader):
    approved={};evidence=[]
    for filename,sha,flag in REVIEWS:
        ref=dict(path=str(Q/filename),sha256=sha);value=reader.load(ref)
        require(value['reviewer']=='/root' and value[flag]is True,'Actual independent root source review')
        evidence.append(ref)
        pairs=[]
        for field in ('source_sha256','sdk_source_sha256'):
            pairs.extend(value.get(field,{}).items())
        entries=value.get('sources',[])
        if isinstance(entries,dict):pairs.extend(entries.items())
        else:pairs.extend((entry['path'],entry['sha256'])for entry in entries)
        for path,wanted in pairs:
            if not Path(path).is_absolute():path=str(ROOT/path)
            if Path(path).suffix!='.py':continue
            require(re.fullmatch('[a-f0-9]{64}',wanted),'Reviewed source digest')
            approved.setdefault(path,{}).setdefault(wanted,[]).append(ref)
    # Native source pins are explicitly retained in the independently accepted
    # diagnostic contract. They are not inferred from today's arbitrary bytes.
    contract_review=dict(path=str(Q/REVIEWS[4][0]),sha256=REVIEWS[4][1])
    for path,sha in provider.contract.NATIVE_SOURCES.items():approved.setdefault(path,{}).setdefault(sha,[]).append(contract_review)
    return approved,evidence


def editable_files(distribution,reader):
    directory=Path(distribution['directory']);direct=directory/'direct_url.json'
    if not direct.exists():return {},None,[]
    raw=reader.raw(direct);value=json.loads(raw)
    if value.get('dir_info',{}).get('editable')is not True:return {},dict(reference=reference(direct,raw),editable=False),[]
    parsed=urlparse(value['url']);require(parsed.scheme=='file' and not parsed.netloc,'Local editable source only')
    root=Path(unquote(parsed.path));require(root==provider.EDITABLE and distribution['name']=='vllm_neuron',
                                          'Unknown editable root cannot change the frozen provider search path')
    record=reader.raw(directory/'RECORD').decode()
    finder_names=[relative for relative,_,_ in csv.reader(record.splitlines())
                  if Path(relative).name.startswith('__editable___') and relative.endswith('_finder.py')]
    require(len(finder_names)==1,'Exact one actual editable finder source')
    finder=SITE/finder_names[0];finder_raw=reader.raw(finder);tree=ast.parse(finder_raw)
    mappings=[]
    for node in tree.body:
        if isinstance(node,ast.AnnAssign)and isinstance(node.target,ast.Name)and node.target.id=='MAPPING':
            mappings.append(ast.literal_eval(node.value))
    require(len(mappings)==1 and mappings[0].get('vllm_neuron')==str(root/'vllm_neuron'),
            'Literal installed editable mapping agrees with exact original provider root')
    files={}
    for path in sorted((root/'vllm_neuron').rglob('*.py')):
        raw=reader.raw(path);files[str(path)]=dict(sha256=digest(raw),bytes=len(raw))
    return files,dict(reference=reference(direct,raw=reader.raw(direct)),root=str(root),finder=reference(finder,finder_raw),
        mapping=mappings[0],selected_runtime_package='vllm_neuron',pth_executed=False,
        unselected_mappings={k:v for k,v in mappings[0].items()if k!='vllm_neuron'}),[]


def saved_elf_table(reader):
    result={};refs=[]
    for ref in ELF_EVIDENCE:
        value=reader.load(ref);refs.append(ref)
        rows=value.get('files',[]) if 'files'in value else value.get('new_elf_objects',[])
        for row in rows:
            reference_value=row.get('reference');elf=row.get('elf')
            if reference_value is None or elf is None:continue
            path=reference_value['path'];entry=dict(sha256=reference_value['sha256'],bytes=reference_value['bytes'],
                elf=elf,evidence=ref)
            if path in result:
                require(all(result[path][key]==entry[key]for key in ('sha256','bytes')),'Saved ELF content identity conflict')
            result[path]=entry
    return result,refs


def reviewed_python_payloads(payloads,reader,approved):
    """Do not turn arbitrary current source drift into a RECORD override."""
    updated=dict(payloads);overrides={};pins={};issues=[];checked=0
    for path,expected in payloads.items():
        if Path(path).suffix!='.py':continue
        try:
            raw=reader.raw(path);actual=dict(sha256=digest(raw),bytes=len(raw));checked+=1
            if actual!=expected:
                if actual['sha256'] in approved.get(path,{}):
                    overrides[path]=actual;pins[path]=actual['sha256'];updated[path]=actual
                else:issues.append(dict(kind='unreviewed_modified_Python_payload',path=path,record=expected,actual=actual))
        except (OSError,ValueError,UnicodeError)as error:issues.append(dict(kind='Python_source_unresolved',path=path,error=str(error)))
    return updated,overrides,pins,issues,checked


def ambient_record_conflicts(inventory,selected,payloads,reader):
    """An unselected installed wheel can still own the exact selected paths."""
    issues=[];records=[];duplicate_owners=[]
    for name in sorted(set(inventory)-set(selected)):
        path=Path(inventory[name]['directory'])/'RECORD'
        try:
            raw=reader.raw(path);entries=record_entries(raw.decode());records.append(reference(path,raw))
        except (OSError,ValueError,binascii.Error,csv.Error)as error:
            issues.append(dict(kind='ambient_RECORD_ownership_unresolved',name=name,error=str(error)));continue
        for target in sorted(set(entries)&set(payloads)):
            owners=dict(path=target,unselected_distribution=name,selected_record=payloads[target],ambient_record=entries[target])
            if payloads[target]!=entries[target]:issues.append(dict(kind='ambient_conflicting_RECORD_owner',**owners))
            else:duplicate_owners.append(owners)
    return dict(records=records,identical_duplicate_owners=duplicate_owners,issues=issues,
        environment_wide_dependency_compatibility_claimed=False,
        actual_ambiguous_file_contents_read=False)


def elf_closure(payloads,saved):
    """Exact original pathname+content only; no cross-prefix/ABI substitution."""
    native={path:row for path,row in payloads.items()if '.so'in Path(path).name}
    matched={};unresolved=[];edges=[];linked={};queue=list(native);seen=set()
    while queue:
        path=queue.pop(0)
        if path in seen:continue
        seen.add(path);expected=payloads.get(path,linked.get(path));row=saved.get(path)
        if row is None or any(row[key]!=expected[key]for key in ('sha256','bytes')):
            unresolved.append(dict(kind='missing_exact_original_ELF_metadata',path=path,expected=expected));continue
        matched[path]=row['evidence']
        if not isinstance(row['elf'].get('needed'),list)or not all(isinstance(n,str)for n in row['elf']['needed']):
            unresolved.append(dict(kind='incomplete_saved_ELF_metadata',path=path));continue
        for needed in row['elf']['needed']:
            candidates=[p for p,data in saved.items()if data['elf'].get('soname',Path(p).name)==needed]
            edge=dict(requester=path,needed=needed,static_saved_candidates=candidates,actual_loader_selection_proven=False)
            edges.append(edge)
            if len(candidates)!=1:
                unresolved.append(dict(kind='unresolved_or_ambiguous_saved_ELF_dependency',**edge));continue
            target=candidates[0]
            # Saved candidate names do not establish visibility or selection,
            # including when the only saved candidate is a system library.
            unresolved.append(dict(kind='saved_ELF_candidate_selection_not_proven',**edge))
            if Path(target).is_relative_to(Path('/usr/lib')) or Path(target).is_relative_to(Path('/lib')):
                edge['host_system_ELF_boundary']=True;continue
            if 'site-packages'in Path(target).parts and not Path(target).is_relative_to(SITE):
                unresolved.append(dict(kind='other_environment_ELF_candidate_rejected',**edge));continue
            if target not in payloads:
                linked[target]={key:saved[target][key]for key in ('sha256','bytes')}
            queue.append(target)
            # A SONAME match supplies an overinclusive prehash candidate, not
            # proof of visibility/selection under the original loader state.
    return linked,dict(native_record_objects=len(native),exact_saved_objects=len(matched),matched=matched,
        edges=edges,unresolved=unresolved,complete=not unresolved,
        other_prefix_saved_objects_not_reused=sum(path not in payloads for path in saved),actual_native_bytes_read=False)


def generate(output):
    reader=TextReader();issues=[]
    cfg=reader.raw(Path('/home/ubuntu/venvs/vllm_neuron/pyvenv.cfg')).decode()
    require(re.search(r'^version\s*=\s*3\.10\.12\s*$',cfg,re.M),'Original fixed Python3.10 metadata')
    inventory=installed_metadata(reader)
    graph=dependency_graph(inventory,provider.REQUIRED,ENV,ROOT_EXTRAS)
    approved,review_refs=reviewed_sources(reader)
    source_map={};payloads={};distributions=[];overrides={};additional={};editable=[];record_owners={}
    for name in sorted(graph['selected']):
        if name not in inventory:continue
        distribution=inventory[name];record=Path(distribution['directory'])/'RECORD'
        try:
            raw=reader.raw(record);record_ref=reference(record,raw);entries=record_entries(raw.decode())
        except (OSError,ValueError,binascii.Error,csv.Error)as error:
            issues.append(dict(kind='unresolved_record',name=name,error=str(error)));continue
        source_map[str(record)]=record_ref['sha256']
        distributions.append(dict(name=distribution['wheel_name'],version=distribution['version'],record=record_ref))
        for path,entry in entries.items():
            record_owners.setdefault(path,[]).append(name)
            if path in payloads and payloads[path]!=entry:
                issues.append(dict(kind='conflicting_RECORD_owners',path=path,owners=record_owners[path]));continue
            payloads[path]=entry
        try:
            files,origin,extra=editable_files(distribution,reader)
            if origin is not None:editable.append(origin)
            for path,entry in files.items():
                if entry['sha256']not in approved.get(path,{}):issues.append(dict(kind='editable_source_not_independently_reviewed',path=path))
                else:additional[path]=entry;source_map[path]=entry['sha256']
            issues.extend(extra)
        except (OSError,ValueError,SyntaxError)as error:issues.append(dict(kind='unresolved_editable',name=name,error=str(error)))
    ambient=ambient_record_conflicts(inventory,graph['selected'],payloads,reader)
    issues.extend(ambient['issues'])
    # Verify Python source changes against existing independent review pins;
    # no new source override is authorized just because current bytes differ.
    payloads,overrides,pins,python_issues,python_checked=reviewed_python_payloads(payloads,reader,approved)
    source_map.update(pins);issues.extend(python_issues)
    # Preserve the root-reviewed direct local source superset, including all
    # lazy historical branches. Do not import history to infer a smaller map.
    local={path for path in approved if Path(path).parent in (ROOT,Q)}|set(provider.contract.NATIVE_SOURCES)
    for path in sorted(local):
        try:
            raw=reader.raw(path);actual=dict(sha256=digest(raw),bytes=len(raw))
            require(actual['sha256']in approved[path],'Current source matches an independent reviewed hash')
            source_map[path]=actual['sha256']
            if path not in payloads:additional[path]=actual
            elif payloads[path]!=actual:issues.append(dict(kind='fixed_source_payload_conflict',path=path))
        except (OSError,ValueError,UnicodeError)as error:issues.append(dict(kind='reviewed_local_source_unresolved',path=path,error=str(error)))
    payloads.update(additional)
    saved,elf_refs=saved_elf_table(reader);linked,elf=elf_closure(payloads,saved);payloads.update(linked)
    manifest=dict(schema='spark26_native_provider_manifest_v1',distributions=distributions,python_overrides=overrides,
        additional_files=additional,linked_native_files=linked,payload_count=len(payloads),
        payload_bytes=sum(row['bytes']for row in payloads.values()),payload_sha256=provider.contract.digest(payloads))
    try:
        # The accepted independent runtime parser is unchanged, not an injected
        # success predicate. It re-reads actual RECORD text, not native payloads.
        reconstructed=provider.merge_manifest(manifest,source_map,lambda ref:reader.raw(ref['path'],ref['sha256']).decode())
        require(reconstructed==payloads,'Independent frozen Runtime merge agrees with every generated row')
        validation=dict(passed=True,payloads_equal=True,validator=str(Path(provider.__file__).resolve()),
                        payload_count=len(reconstructed),payload_bytes=manifest['payload_bytes'])
    except (ValueError,OSError)as error:validation=dict(passed=False,error=str(error),validator=str(Path(provider.__file__).resolve()))
    issues.append(dict(kind='dynamic_import_closure_not_proven_by_declared_metadata',
        explanation='METADATA, Python review pins and saved ELF metadata do not prove all lazy imports or dlopen targets'))
    preparation=dict(schema='spark26_native_manifest_preparation_v1',created_at=datetime.now(timezone.utc).isoformat(),
        target_python=provider.contract.PYTHON,target_environment=ENV,environment_is_pinned_target_not_current_runtime_observation=True,
        original_roots=provider.REQUIRED,root_extras=ROOT_EXTRAS,installed_distribution_count=len(inventory),
        selected_distribution_count=len(graph['selected']),graph=graph,editable=editable,review_evidence=review_refs,
        python_source_files_checked=python_checked,python_override_provenance={p:approved[p][row['sha256']]for p,row in overrides.items()},
        local_source_policy='Complete independently reviewed ROOT/Q direct .py superset plus fixed official/native source pins',
        saved_elf_evidence=elf_refs,elf=elf,ambient_record_ownership=ambient,issues=issues,independent_runtime_merge=validation,
        complete_runtime_closure=not graph['unresolved']and not issues and elf['complete']and validation['passed'],
        actual_native_hash_authentication=False,actual_native_imports=False,weights_read=False,
        native_payload_bytes_read=0,packages_changed=False,runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False,
        runtime_authentication_budget_bytes=provider.MAX_RUNTIME_AUTHENTICATION_READ,total_runtime_IO_limit_claimed=False,
        current_text_read_bytes=reader.bytes,text_input_count=len(reader.references))
    output=Path(output)
    require(output.parent==Q and re.fullmatch('spark26-native-manifest-candidate-[a-z0-9-]+',output.name)
            and not output.exists(),'New bounded source candidate namespace only')
    output.mkdir(mode=0o700)
    artifacts={}
    for filename,value in [('manifest.json',manifest),('source-map.json',source_map),('preparation.json',preparation),
                           ('text-inputs.json',reader.references),('selected-distributions.json',
                            {name:inventory[name]for name in graph['selected']if name in inventory})]:
        raw=encoded(value);require(len(raw)<=64*2**20,'Bounded candidate metadata output')
        with (output/filename).open('xb')as stream:stream.write(raw)
        artifacts[filename]=reference(output/filename,raw)
    return dict(artifacts=artifacts,selected_distributions=len(distributions),payload_count=manifest['payload_count'],
        payload_bytes=manifest['payload_bytes'],python_overrides=len(overrides),additional_sources=len(additional),
        dependency_issues=len(graph['unresolved']),source_issues=len(issues),ELF_issues=len(elf['unresolved']),
        independent_runtime_merge=validation,complete_runtime_closure=preparation['complete_runtime_closure'],runtime_admitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(generate(args.output),indent=2))
