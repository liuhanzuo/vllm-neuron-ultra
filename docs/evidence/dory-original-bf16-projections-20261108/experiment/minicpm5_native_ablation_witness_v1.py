"""Diagnostic full-vocabulary evidence; runtime-only tensor operations.

No file IO or native import. Injected authenticated writer must durably save
four raw vectors and one frame report, including failures. Equality stays exact.
"""
import math
from minicpm5_layer0_capture_endpoints_v1 import require,tensor_digest
VOCAB=130560
VECTOR_BYTES=VOCAB*12


def metrics(torch,actual,expected):
    finite_a=torch.isfinite(actual);finite_b=torch.isfinite(expected)
    bad_a=int((~finite_a).sum());bad_b=int((~finite_b).sum())
    equal=int((actual==expected).sum())
    out=dict(equal_elements=equal,differing_elements=VOCAB-equal,actual_nonfinite=bad_a,reference_nonfinite=bad_b)
    if bad_a or bad_b:
        out.update(finite=False,max_abs=None,mean_abs=None,rmse=None,relative_l2=None,cosine=None,kl_reference_to_actual=None,
            actual_argmax=None,reference_argmax=None,actual_margin=None,reference_margin=None)
        return out
    a=actual.double();b=expected.double();delta=a-b
    norm=float(torch.linalg.vector_norm(b));top_a=torch.topk(a,2).values;top_b=torch.topk(b,2).values
    loga=torch.log_softmax(a,dim=-1);logb=torch.log_softmax(b,dim=-1)
    out.update(finite=True,max_abs=float(delta.abs().max()),mean_abs=float(delta.abs().mean()),rmse=float((delta.square().mean()).sqrt()),
        relative_l2=float(torch.linalg.vector_norm(delta))/max(norm,1e-300),cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
        kl_reference_to_actual=float((logb.exp()*(logb-loga)).sum()),actual_argmax=int(a.argmax()),reference_argmax=int(b.argmax()),
        actual_margin=float(top_a[0]-top_a[1]),reference_margin=float(top_b[0]-top_b[1]))
    require(all(not isinstance(value,float) or math.isfinite(value) for value in out.values()),'Finite diagnostic metrics')
    return out


def record(torch,bf16,expected,identity,writer):
    require(list(bf16.shape)==list(expected.shape)==[VOCAB] and bf16.dtype==torch.bfloat16 and expected.dtype==torch.float32,'Original full-vocabulary witness shape/dtype')
    require(bf16.device.type==expected.device.type=='cpu','CPU witness only')
    actual=bf16.detach().float().cpu().clone();reference=expected.detach().cpu().clone();reference_bf16=reference.to(torch.bfloat16)
    # The original BF16 reference was stored as FP32; verify that derived BF16
    # is a reversible representation, never use it to relax the original gate.
    ref_hash=tensor_digest(torch,reference);actual_hash=tensor_digest(torch,actual)
    require(tensor_digest(torch,reference_bf16.float())==ref_hash,'Reference no longer exact BF16-to-FP32 representation')
    vectors=[('actual-bf16',bf16.detach().clone()),('actual-fp32',actual),('expected-fp32',reference),('expected-bf16-derived',reference_bf16)]
    refs=[]
    for name,value in vectors:
        value=value.contiguous();metadata=dict(shape=list(value.shape),stride=list(value.stride()),offset=value.storage_offset(),dtype=str(value.dtype),
            bytes=value.numel()*value.element_size(),sha256=tensor_digest(torch,value),identity=identity,derived=name=='expected-bf16-derived')
        ref=writer.tensor(name,value,metadata)
        require(ref['sha256']==metadata['sha256'] and ref['bytes']==metadata['bytes'],'Writer must durably verify exact witness bytes')
        refs.append(dict(name=name,metadata=metadata,reference=ref))
    require(sum(row['metadata']['bytes'] for row in refs)==VECTOR_BYTES,'All four full-vocabulary witnesses')
    exact=actual_hash==ref_hash
    report=dict(schema='minicpm5_native_ablation_frame_witness_v1',identity=identity,passed=exact,raw_exact=exact,
        diagnostic_observed=True,original_full_vocab=VOCAB,tensors=refs,metrics=metrics(torch,actual,reference),
        actual_fp32_sha256=actual_hash,original_expected_fp32_sha256=ref_hash,
        original_gate_failure=None if exact else 'Original native32 logits changed',full_model_qualified=False,new_speedup_claim=False)
    saved=writer.record('witness',report)
    require(saved['durable'] is True,'Witness receipt must be durable before gate result')
    return report,saved
