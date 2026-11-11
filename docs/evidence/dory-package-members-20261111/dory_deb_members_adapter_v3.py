"""Download six signed archives or independently replay saved bytes; no install."""
from pathlib import Path
import dory_serial_stage_contract_v2 as c
import dory_deb_input_binding_v3 as binding
import dory_deb_transport_v3 as io


def run(guard,emit,trust,out,mode,prior_ref=None):
    c.require(trust==binding.TRUST and mode in ('download','replay')and (prior_ref is None)==(mode=='download'),'Exact reviewed archive mode')
    n,authority=binding.authenticated(guard);tool=authority['tool'];guard();Path(out).mkdir(mode=0o700)
    prior=c.load(prior_ref,8*c.MIB)if mode=='replay'else None
    if prior is not None:c.require(prior['schema']=='dory_deb_members_result_v1'and prior['complete']and prior['inputs']==trust and len(prior['packages'])==6,'Exact full prior package result')
    packages=[];stream_total=originals=current=0
    for index,row in enumerate(n['packages']):
        guard()
        if mode=='download':
            archive=Path(out)/(row['package']+'.deb');io.download(row,archive,guard,emit)
        else:
            previous=prior['packages'][index];c.require(previous['package']==row['package']and previous['version']==row['version'],'Ordered prior package identity')
            archive=Path(previous['inspection']['archive']['path']);io.file_ref(archive,guard,row)
            emit(dict(event='archive_replay_start',package=row['package'],archive=previous['inspection']['archive']))
        inspection=io.inspect(row,archive,tool,guard,emit);stream_total+=inspection['tar']['stream_bytes'];c.require(stream_total<=512*c.MIB,'Six-package aggregate decompressed stream cap')
        comparisons=[]
        for target in row['targets']:
            guard();member=inspection['tar']['targets'][target['observed_path']];expected=dict(bytes=member['size'],sha256=member['sha256'])
            old=target['original_observed_reference'];oldmatch=None
            if old is not None:
                oldmatch=all(old[k]==expected[k]for k in ('bytes','sha256'));c.require(oldmatch,'Original saved library bytes equal authenticated archive member');originals+=1
            host=target['observed_path'];links=2 if host in ('/usr/lib/x86_64-linux-gnu/libc.so.6','/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30')else 1
            observed,identity=io.file_ref(host,guard,expected,expected_nlink=links)
            current+=1;comparisons.append(dict(target=target,member=dict(bytes=member['size'],sha256=member['sha256'],terminal=member['terminal']),original_matches=oldmatch,current=observed,current_identity=identity,current_matches=True,new_observation_only=old is None))
        package=dict(package=row['package'],version=row['version'],inspection=inspection,comparisons=comparisons)
        if prior is not None:c.require(package==prior['packages'][index],'Independent complete package replay equals original member/alias/tool/current results')
        packages.append(package);emit(dict(event='package_members_verified',package=row['package'],targets=len(comparisons),tar_sha256=inspection['tar']['stream_sha256']))
    c.require(originals==11 and current==12,'All original/current targets verified')
    value=dict(schema='dory_deb_members_result_v1',complete=True,mode=mode,replay_from=prior_ref,inputs=trust,packages=packages,archive_total_bytes=n['archive_total_bytes'],decompressed_total_bytes=stream_total,target_count=12,original_match_count=originals,current_match_count=current,package_content_authenticated=True,loader_selection_established=False,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False)
    result=c.save(Path(out)/'sources-result.json',value,8*c.MIB);guard();emit(dict(event='six_packages_complete',mode=mode,result=result,packages=6,targets=12,original_matches=11,current_matches=12))
    return dict(result=result,packages=6,targets=12,mode=mode)
