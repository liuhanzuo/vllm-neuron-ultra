"""Independent stdlib verification of complete upstream/contrast saved bytes.

No launch, native imports or original norm/attention math. Caller must provide
actual post-exit serial admission. BF16 residual adds are rederived exactly.
"""
from contextlib import ExitStack
import hashlib,json,os,stat,struct
from pathlib import Path
import review_minicpm5_frame1_raw_boundaries_v1 as old
require,same,Metric,VALUES,compare_metric=old.require,old.same,old.Metric,old.VALUES,old.compare_metric
ROLES=('native','cpu-rank0','cpu-rank1')
FIELDS=('layer_input','inputnorm_output','attention_input','attention_output','postnorm_input','postnorm_output','mlp_input')
NORMS=('input_layernorm','post_attention_layernorm')
CONTRASTS=('native_entry_only','native_attention_only','native_entry_and_attention')
LENGTH,BUCKET,WIDTH=1231,2048,2048
OLD_PIN='22a5e192c31ad6fe838e5ab9bb6e274e982b2a3d1239c1776d934bb2ba5d0895'
ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
R3_REFS={role:dict(path=str(ROOT/'minicpm5-frame1-mlp-r3-20261108-v1'/role/'capture.json'),sha256=sha)for role,sha in zip(ROLES,('296c9a721f17c3118d18cb3cc9f929e648eb9d673c29e69aedf227b192d16348','38c8ba9a5e11ec366ab570e908f0b56e9a6e8fb389cd142cb7637ba94c7a0ca3','9f18c3163775cdbab4cb5789758d61f63472b0b3aea2dfd370e3d123d7908c45'))}
CAPTURE_PLAN=dict(schema='minicpm5_upstream_contrast_capture_plan_v1',contrasts=list(CONTRASTS),fields=['postnorm_output'],shape=[1,2048,2048],dtype='torch.bfloat16',tensor_bytes=8388608,contrast_bytes=25165824,all_raw_bytes=177926144,whole_output_cap_bytes=201326592,independent_norm_arithmetic_reexecuted=False,new_speedup_claim=False)


def bf16_add(left,right):
    a,b=VALUES[left],VALUES[right];require(a is not None and b is not None,'Finite residual operands')
    value=a+b
    if value==0:return 0x8000 if left==right==0x8000 else 0
    sign=0x8000 if value<0 else 0;value=abs(value)
    if value<128:return sign|value
    shift=max(0,value.bit_length()-8);mantissa=value>>shift
    remainder=value-(mantissa<<shift)
    if shift and (remainder>(1<<(shift-1))or(remainder==(1<<(shift-1))and mantissa&1)):mantissa+=1
    if mantissa==256:mantissa=128;shift+=1
    exponent=shift+1
    if exponent>=255:return sign|0x7f80
    return sign|(exponent<<7)|(mantissa-128)


class TensorFile:
    def __init__(self,row,path,shape,kind,guard):
        self.row,self.path,self.guard=row,Path(path),guard;guard()
        same(row['path'],str(self.path),'Exact tensor path');require(self.path.is_absolute()and self.path.resolve()==self.path and not any(p.is_symlink()for p in (self.path,*self.path.parents)),'Canonical tensor path')
        if kind=='observed':require(row['observed_not_recomputed']is True and row['encoding']=='packed_bf16_le','Actual model observation')
        else:require(kind=='computed'and row['counterfactual_computed']is True and row['observed_model_output']is False and row['encoding']=='packed_bf16_le','Explicit computed counterfactual, never observed model output')
        size=old.meta.layout(row['metadata'],shape);require(size==row['bytes']==row['identity']['size']and 0<size<=16*2**20,'Complete finite tensor size')
        same(row['sha256'],row['metadata']['sha256'],'Metadata/raw linkage')
        self.fd=os.open(self.path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            info=os.fstat(self.fd);require(stat.S_ISREG(info.st_mode)and info.st_nlink==1 and info.st_uid==os.getuid(),'Owned regular tensor')
            same(old.identity(self.path,info),row['identity'],'Original tensor identity')
        except BaseException:os.close(self.fd);raise
        self.length,self.width=shape[1:];self.index=0;self.digest=hashlib.sha256()
        self.nonzero=dict(full=False,real_tokens=False,prediction=False)
    def __enter__(self):return self
    def __exit__(self,*args):os.close(self.fd)
    def next_row(self):
        if self.index%64==0:self.guard()
        require(self.index<self.length,'Full bounded row range');raw=os.pread(self.fd,2*self.width,2*self.width*self.index)
        same(len(raw),2*self.width,'Complete row bytes');self.digest.update(raw);words=struct.unpack('<'+str(self.width)+'H',raw)
        require(all(VALUES[x]is not None for x in words),'Finite complete tensor')
        nz=any(x&0x7fff for x in words);self.nonzero['full']|=nz
        if self.index<LENGTH:self.nonzero['real_tokens']|=nz
        if self.index==LENGTH-1:self.nonzero['prediction']|=nz
        self.index+=1;return words
    def self_metrics(self):
        require(self.index==self.length,'All original rows observed')
        def result(n,nz):return dict(raw_exact=True,different_elements=0,compared_elements=n,nonfinite_elements=0,max_abs=0.0,mean_abs=0.0,cosine=1.0 if nz else None)
        return {scope:result(count*self.width,self.nonzero[scope])for scope,count in [('full',self.length),('real_tokens',min(self.length,LENGTH)),('prediction',1)]}
    def finish(self):
        same(self.index,self.length,'Every original row read');same(self.digest.hexdigest(),self.row['sha256'],'Full tensor bytes authenticated')
        digest=hashlib.sha256()
        for offset in range(0,self.row['bytes'],65536):
            self.guard();part=os.pread(self.fd,min(65536,self.row['bytes']-offset),offset);digest.update(part)
        same(digest.hexdigest(),self.row['sha256'],'Tensor unchanged at completion')
        same(old.identity(self.path,os.fstat(self.fd)),self.row['identity'],'Opened tensor stable');same(old.identity(self.path,self.path.lstat()),self.row['identity'],'Path stable')
        return dict(path=str(self.path),sha256=digest.hexdigest(),bytes=self.row['bytes'])


def observed_row(report,field):
    rows=[row for row in report['captures']if row['name']==field];same(len(rows),1,'Unique original field');return rows[0]


def open_observed(stack,root,reports,role,field,guard):
    return stack.enter_context(TensorFile(observed_row(reports[role],field),root/role/(field+'.bf16'),[1,LENGTH if role=='native'else BUCKET,WIDTH],'observed',guard))


def audit_boundaries(root,reports,replay,guard):
    references=[];metrics={};self_metrics={role:{}for role in ROLES}
    for field in FIELDS:
        with ExitStack()as stack:
            streams=[open_observed(stack,root,reports,role,field,guard)for role in ROLES]
            real,prediction=Metric(),Metric()
            for index in range(BUCKET):
                a,b=streams[1].next_row(),streams[2].next_row();same(a,b,'Both complete replicated CPU boundaries')
                if index<LENGTH:
                    native=streams[0].next_row();real.add(native,a)
                    if index==LENGTH-1:prediction.add(native,a)
            metrics[field]=dict(real_tokens=real.result(),prediction=prediction.result())
            for role,stream in zip(ROLES,streams):self_metrics[role][field]=stream.self_metrics();references.append(stream.finish())
        if 'observed_boundaries'in replay:
            for scope in ('real_tokens','prediction'):compare_metric(metrics[field][scope],replay['observed_boundaries'][field][scope])
    same(len(references),21,'All21original boundaries');same(sum(row['bytes']for row in references),152735744,'Original complete raw byte count')
    return dict(references=references,metrics=metrics,self_metrics=self_metrics)


def audit_residuals(root,reports,guard):
    out={}
    for role in ROLES:
        with ExitStack()as stack:
            a,b,summed=[open_observed(stack,root,reports,role,field,guard)for field in ('layer_input','attention_output','postnorm_input')]
            different=0
            for index in range(a.length):
                left,right,expected=a.next_row(),b.next_row(),summed.next_row()
                actual=tuple(bf16_add(x,y)for x,y in zip(left,right));different+=sum(x!=y for x,y in zip(actual,expected))
            out[role]=dict(different_elements=different,compared_elements=a.length*a.width,exact=different==0)
            require(different==0,'Exact independently rederived BF16 residual sum')
            for stream in (a,b,summed):stream.finish()
    return out


class Scopes:
    def __init__(self):self.full,self.real,self.prediction=Metric(),Metric(),Metric()
    def add(self,a,b,index):
        self.full.add(a,b)
        if index<LENGTH:self.real.add(a,b)
        if index==LENGTH-1:self.prediction.add(a,b)
    def result(self):return dict(full=self.full.result(),real_tokens=self.real.result(),prediction=self.prediction.result())


def audit_contrast(root,reports,label,claimed,guard):
    require(label in CONTRASTS,'Only reviewed local interventions');same(claimed['complete_shape'],[1,BUCKET,WIDTH],'Complete original CPU geometry');same(claimed['only_real_prefix_substituted'],LENGTH,'Only original real prefix');require(claimed['cpu_padding_preserved']is True,'Original CPU padding retained')
    with ExitStack()as stack:
        streams={key:open_observed(stack,root,reports,role,field,guard)for key,role,field in [('ce','cpu-rank0','layer_input'),('ca','cpu-rank0','attention_output'),('cr','cpu-rank0','postnorm_input'),('cp','cpu-rank0','postnorm_output'),('ne','native','layer_input'),('na','native','attention_output'),('nr','native','postnorm_input'),('np','native','postnorm_output')]}
        output=stack.enter_context(TensorFile(claimed['saved_postnorm_output'],root/'contrasts'/(label+'.postnorm.bf16'),[1,BUCKET,WIDTH],'computed',guard))
        residual,post=Scopes(),Scopes();native_residual,native_post=Metric(),Metric();eh,ah=hashlib.sha256(),hashlib.sha256()
        padding_same=True
        for index in range(BUCKET):
            rows={key:stream.next_row()for key,stream in streams.items()if index<stream.length};actual=output.next_row()
            entry=rows['ne']if index<LENGTH and label in ('native_entry_only','native_entry_and_attention')else rows['ce']
            attention=rows['na']if index<LENGTH and label in ('native_attention_only','native_entry_and_attention')else rows['ca']
            eh.update(struct.pack('<'+str(WIDTH)+'H',*entry));ah.update(struct.pack('<'+str(WIDTH)+'H',*attention))
            summed=tuple(bf16_add(x,y)for x,y in zip(entry,attention));residual.add(summed,rows['cr'],index);post.add(actual,rows['cp'],index)
            if index<LENGTH:native_residual.add(summed,rows['nr']);native_post.add(actual,rows['np'])
            else:padding_same&=entry==rows['ce']and attention==rows['ca']and actual==rows['cp']
        require(padding_same,'Actual saved counterfactual padding remains original')
        for key,digest,field in [('entry',eh,'layer_input'),('attention',ah,'attention_output')]:
            expected=dict(observed_row(reports['cpu-rank0'],field)['metadata'],sha256=digest.hexdigest());same(claimed[key],expected,'Exact restored geometry and substituted complete inputs')
        computed=dict(residual_vs_cpu=residual.result(),postnorm_vs_cpu=post.result(),residual_real_vs_native=native_residual.result(),postnorm_real_vs_native=native_post.result())
        for name in ('residual_vs_cpu','postnorm_vs_cpu'):
            for scope in ('full','real_tokens','prediction'):compare_metric(computed[name][scope],claimed[name][scope])
        for name in ('residual_real_vs_native','postnorm_real_vs_native'):compare_metric(computed[name],claimed[name])
        for stream in streams.values():stream.finish()
        reference=output.finish()
    return dict(output=reference,metrics=computed,whole_padding_exact=True,residual_math_independently_reexecuted=True,norm_math_independently_reexecuted=False)


def load_json(reference):
    path=Path(reference['path']);require(path.is_absolute()and path.resolve()==path and not path.is_symlink(),'Canonical metadata')
    with path.open('rb')as stream:raw=stream.read(2*2**20+1)
    require(len(raw)<=2*2**20 and hashlib.sha256(raw).hexdigest()==reference['sha256'],'Bounded exact JSON reference');return json.loads(raw)


def verify_norm_weights(root,reports,guard):
    rows=[];original={}
    for role in ROLES:
        same(set(reports[role]['norm_weights']),set(NORMS),'Both original norms')
        for name in NORMS:
            guard();row=reports[role]['norm_weights'][name];path=root/role/(name+'.weight.bf16')
            same(row['path'],str(path),'Exact norm weight path');require(path.resolve()==path and not path.is_symlink(),'Canonical norm weight');same(row['shape'],[WIDTH],'Original whole norm width');same(row['dtype'],'torch.bfloat16','Original norm dtype')
            with path.open('rb')as stream:raw=stream.read(4097)
            require(len(raw)==row['bytes']==4096 and hashlib.sha256(raw).hexdigest()==row['sha256'],'Whole saved norm weight bytes')
            require(all(VALUES[x]is not None for x in struct.unpack('<2048H',raw)),'Finite whole norm weights')
            if role=='native':original[name]=raw
            else:same(raw,original[name],'Both CPU ranks match native norm weights')
            rows.append(dict(path=str(path),sha256=row['sha256'],bytes=4096))
    return rows


def verify_r3_anchors(reports,guard):
    refs=[]
    for role in ROLES:
        guard();original=load_json(R3_REFS[role]);native=role=='native';rank=-1 if native else int(role[-1])
        require(original['complete']is True and original['native']is native and original['rank']==rank,'Actual prior r3 role')
        same(original['frame'],dict(index=1,fixture=7,slot=1,step=0),'Same original r3 frame')
        require(original['actual_prefix_calls']==1 and original['precision_interventions']==[]and original['anchor']['all_six_exact']is True,'Same original r3 prefix')
        same(observed_row(original,'input')['metadata'],observed_row(reports[role],'mlp_input')['metadata'],'Independent unchanged full r3 MLP input')
        refs.append(R3_REFS[role])
    return refs


def verify_reports(reports,replay):
    same(set(reports),set(ROLES),'All three actual captures');same(replay['schema'],'minicpm5_frame1_upstream_replay_v2','Full-output replay v2 required')
    same(replay['frame'],dict(index=1,fixture=7,slot=1,step=0),'Original replay frame');same(replay['contrast_capture_plan'],CAPTURE_PLAN,'Complete original output budget')
    require(replay['complete']is True and replay['full_model_qualified']is False and replay['unique_final_KL_cause_proven']is False and replay['new_speedup_claim']is False,'Exact local-only outcome')
    for role,report in reports.items():
        native=role=='native';rank=-1 if native else int(role[-1]);require(report['schema']=='minicpm5_frame1_upstream_capture_v1'and report['complete']is True and report['native']is native and report['rank']==rank,'Actual complete capture role')
        anchor=report['cpu_full_anchor'];same(anchor['reference'],R3_REFS[role],'Exact r3 reference');require(anchor['full_mlp_input_exact']is True and anchor['adjacent_boundaries_exact']is True and anchor['source_shape_preserved']is True,'All actual full-input anchors')
        same(report['frame'],dict(index=1,fixture=7,slot=1,step=0),'Original first frame');same([row['name']for row in report['captures']],list(FIELDS),'All original seven boundaries')
        require(report['precision_interventions']==[]and report['oproj_witnesses']==[]and report['actual_prefix_calls']==1 and report['anchor']['all_six_exact']is True,'Unmodified anchored prefix')
        require(report['history_consumed']is False and report['downstream_layers_executed']==0 and report['full_model_qualified']is False and report['new_speedup_claim']is False,'No full-model outcome')
        for left,right in [('inputnorm_output','attention_input'),('postnorm_output','mlp_input')]:same(observed_row(report,left)['metadata'],observed_row(report,right)['metadata'],'Complete adjacent boundary equality')
    same(set(replay['controls']),set(ROLES),'Three complete replay controls')
    fields=('inputnorm_output','attention_input','postnorm_input','postnorm_output','mlp_input');all_exact=True
    for role,controls in replay['controls'].items():
        same(set(controls),set(fields),'All own-input norm/residual controls')
        for field,control in controls.items():
            expected=observed_row(reports[role],field)['metadata'];same(control['expected'],expected,'Control expected actual saved capture')
            require(type(control['raw_exact'])is bool and type(control['layout_exact'])is bool,'Control booleans');exact=control['actual']['sha256']==expected['sha256'];layout=all(control['actual'][k]==expected[k]for k in ('shape','stride','offset','dtype'))
            same(control['raw_exact'],exact,'Actual own-output hash classification');same(control['layout_exact'],layout,'Actual own-output layout classification');all_exact&=exact and layout
    same(replay['all_own_input_controls_exact'],all_exact,'No promoted replay control')
    same(set(replay['contrasts']),set(CONTRASTS)if all_exact else set(),'All three contrasts only after exact controls')
    require(type(replay['passed'])is bool and type(replay['all_own_input_controls_exact'])is bool,'Actual replay outcome booleans');same(replay['passed'],all_exact,'Actual replay outcome classification')
    if all_exact:
        same(set(replay['observed_boundaries']),set(FIELDS),'No omitted observed comparison')
        same(replay['classification'],'local_residual_and_postnorm_contrasts_only_not_attention_or_final_logit_cause','Local successful replay only')
    else:same(replay['classification'],'own_input_replay_failed_no_counterfactual_executed','Retain failed replay control classification')
    return all_exact


def review(*,root,capture_refs,replay_ref,guard):
    require(callable(guard),'Actual independently admitted serial guard');guard();same(hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest(),OLD_PIN,'Original independent metric implementation');same(hashlib.sha256(Path(old.meta.__file__).read_bytes()).hexdigest(),old.METADATA_SHA,'Original independent metadata helper')
    root=Path(root);require(root.is_absolute()and root.resolve()==root,'Canonical actual output root')
    same(set(capture_refs),set(ROLES),'Complete references');reports={role:load_json(ref)for role,ref in capture_refs.items()};replay=load_json(replay_ref)
    same(replay_ref['path'],str(root/'upstream-replay.json'),'Exact actual replay path');same(replay['captures'],capture_refs,'Same original actual capture refs')
    for role,ref in capture_refs.items():same(ref['path'],str(root/role/'capture.json'),'Exact role capture metadata')
    exact=verify_reports(reports,replay);r3=verify_r3_anchors(reports,guard);boundaries=audit_boundaries(root,reports,replay,guard);norms=verify_norm_weights(root,reports,guard);residuals=audit_residuals(root,reports,guard)
    contrasts={}
    if exact:
        for role,controls in replay['controls'].items():
            for field,control in controls.items():
                for scope in ('full','real_tokens','prediction'):compare_metric(boundaries['self_metrics'][role][field][scope],control['comparison'][scope])
        for label in CONTRASTS:contrasts[label]=audit_contrast(root,reports,label,replay['contrasts'][label],guard)
        same(sum(value['output']['bytes']for value in contrasts.values()),25165824,'Three complete saved counterfactual outputs')
    guard();return dict(schema='minicpm5_upstream_independent_saved_bytes_review_v1',complete=True,replay=replay_ref,captures=capture_refs,r3_full_input_anchors=r3,original_boundary_files=boundaries['references'],original_boundary_metrics=boundaries['metrics'],norm_weight_files=norms,residual_checks=residuals,contrasts=contrasts,all21_original_payloads_authenticated=True,all6_norm_weight_files_authenticated=True,all3_contrast_payloads_authenticated=exact,all_observed_metrics_independently_recomputed=True,residual_addition_independently_reexecuted=True,norm_arithmetic_independently_reexecuted=False,attention_independently_reexecuted=False,replay_controls_passed=exact,actual_exit_verified=False,physical_drain_verified=False,full_model_qualified=False,new_speedup_claim=False)
