"""Pure source renderer for the accepted r2 helper lineage; never bindings/IO.

render(sources) accepts three complete Python strings and returns three Python
strings. It imports no experiment runtime and performs no filesystem reads or
writes. Source authentication and occurrence counts fail closed.
"""
import ast
import hashlib

BASE_SHA256={
    'prepare_qwen35_kv256_alias_diagnostic_binding_v2.py':'ce9174209b21412dde68a1b6d08e91394459543d6d6fafa89df23dbcd37ac98e',
    'qwen35_kv256_alias_diagnostic_root_helpers_v1.py':'3128db4d230d614111535d5aa7e6baa03eea2b230fad11775ced873b14153520',
    'qwen35_kv256_alias_diagnostic_audit_v1.py':'c3770120b3ad73aee92708acea8ff182338ae41a3db65306be98b420863f3a49',
}
OLD='qwen35_kv256_alias_diagnostic'
NEW='qwen35_kv256_s16_pair'


def exact(text,before,after,count=1):
    if text.count(before)!=count:raise ValueError('Exact reviewed source occurrence: '+before)
    return text.replace(before,after)


def namespace(text):
    return text.replace(OLD,NEW).replace('qwen35-kv256-alias-diagnostic-','qwen35-kv256-s16-pair-')


def render(sources):
    if set(sources)!=set(BASE_SHA256):raise ValueError('Exactly three accepted r2 helper source strings')
    for name,text in sources.items():
        if type(text)is not str or hashlib.sha256(text.encode()).hexdigest()!=BASE_SHA256[name]:
            raise ValueError('Changed frozen helper source: '+name)
    # The one inherited plan and frozen v1 preparer remain historical identities.
    before="PREDECESSOR_PREPARER = dict(path=str(Q/'prepare_qwen35_kv256_alias_diagnostic_binding_v1.py'),\n    sha256='d903881e0a66e0d9e5baec131c7974d08d16935ebdf69439ac2576032f6c5c79')"
    preparer=exact(sources['prepare_qwen35_kv256_alias_diagnostic_binding_v2.py'],before,'PREDECESSOR_PREPARER_MARKER')
    preparer=namespace(preparer)
    preparer=exact(preparer,'PREDECESSOR_PREPARER_MARKER',before)
    preparer=exact(preparer,'prepare_qwen35_kv256_s16_pair_binding_v2.py','prepare_qwen35_kv256_s16_pair_binding_v1.py')
    preparer=exact(preparer,'qwen35_kv256_s16_pair_binding_source_delivery_v2','qwen35_kv256_s16_pair_binding_source_delivery_v1',count=2)
    helpers=namespace(sources['qwen35_kv256_alias_diagnostic_root_helpers_v1.py'])
    helpers=exact(helpers,'zip(counts,(1,24,8))','zip(counts,(1,6,2))')
    helpers=exact(helpers,'counts==(1,24,8)','counts==(1,6,2)')
    auditor=namespace(sources['qwen35_kv256_alias_diagnostic_audit_v1.py'])
    auditor=exact(auditor,"report=None, summary=None, artifacts=artifacts, compiled_graphs=len(artifacts),",
        "report=None, summary=None, artifacts=artifacts, compiled_graphs=len(artifacts), artifact_receipts=len(artifacts),")
    auditor=exact(auditor,"                incomplete_proofs=saved['incomplete_proofs'],",
        "                incomplete_proofs=saved['incomplete_proofs'], compiled_graphs=saved['graphs'],\n"
        "                graphs_without_receipt=saved['graphs_without_receipt'],\n"
        "                last_saved_stage=saved['last_saved_stage'], pending_hlo_proof=saved['pending_hlo_proof'],")
    auditor=exact(auditor,'Artifacts emitted before report updates still form the original compile prefix',
        'Completed HLO artifact receipts form an ordered prefix; final graph may precede its receipt')
    result={
        'prepare_qwen35_kv256_s16_pair_binding_v1.py':preparer,
        'qwen35_kv256_s16_pair_root_helpers_v1.py':helpers,
        'qwen35_kv256_s16_pair_audit_v1.py':auditor,
    }
    for name,text in result.items():ast.parse(text,filename=name)
    return result
