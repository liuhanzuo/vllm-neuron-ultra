"""Same actual complete v6 argv; admission hash names its versioned wire bytes."""
from research import minicpm5_sdk_mapping_nocache_launch_v1 as previous
from research import minicpm5_formal_literal_codec_v1 as codec

require = previous.require
encoded = previous.encoded
reference = previous.reference
launch = previous.launch
separate_authorities = previous.separate_authorities


def admitted_sdk(entry_bytes, entry_ref, admission, admission_ref, outer_deadline):
    reference(admission_ref)
    require(entry_ref == {key: codec.LITERAL[key] for key in ('path', 'sha256')}, 'Unchanged pinned v6 entry reference')
    literal = entry_bytes.decode('utf-8')
    raw = codec.encode('admission', admission, literal)
    require(codec.digest(raw) == admission_ref['sha256'], 'Actual canonical wire admission SHA256')
    plan = admission['source_entry_plan']
    require(plan['role'] == 'sdk310' and plan['successor_input'] == 'admission:self'
            and admission['source_entry_binding']['outer_deadline'] == outer_deadline,
            'Same original actual parent admission and absolute deadline')
    plan_ref = dict(path=admission_ref['path'], sha256=codec.digest(encoded(plan)))
    argv = launch(entry_bytes, entry_ref, plan, plan_ref, outer_deadline=outer_deadline)
    argv[11] = admission_ref['sha256']
    return argv
