"""Exact inherited metadata/encoding exceptions; never reads model payloads."""
from pathlib import Path
import hashlib
import json
import nemotron9b_package_metadata_v10 as inherited

# Exact original wheel installation receipts: authenticated by retained wheel.json.
# No general extensionless-file exception and no model/native payload reads.
INSTALLATION_RECEIPTS = {
    '/mnt/nvme/smollm3-cache-ablation-native-v3/target/vllm_neuron-0.2.2+lts.dist-info/REQUESTED':
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    '/mnt/nvme/smollm3-cache-ablation-native-v3/target/vllm_neuron-0.2.2+lts.dist-info/INSTALLER':
        'ceebae7b8927a3227e5303cf5e0f1f7b34bb542ad7250ac03fbcde36ec2f1508',
}

# This completed catalog exceeds the ordinary8MiB cap. Only this frozen exact
# path, digest and complete-catalog schema may use the original16MiB writer cap.
COMPLETED_CATALOGS = {
    '/home/ubuntu/experiments/trn1-latest-models-20261031/research/root-post-smol-sio12-complete-predecessors-20261108-v1.json':
        '958a913c7b32594a1f8231bcbff07cdf194312301f33244a0eca4cc38cf1ab2a',
}

# The inherited reader enumerates named/schema-checked 32MiB Nemotron envelopes.
# Everything else keeps its ordinary8MiB cap and text type restrictions.
def source_bytes(path,wanted=None):
    path=Path(path);name=str(path)
    if name in COMPLETED_CATALOGS:
        expected=COMPLETED_CATALOGS[name]
        inherited.require(wanted is None or wanted==expected,'Exact completed catalog pin')
        inherited.low.clear_window()
        raw=inherited.stable_bytes(path,16*2**20)
        inherited.require(hashlib.sha256(raw).hexdigest()==expected,'Completed catalog bytes changed')
        value=json.loads(raw.decode('utf8'))
        inherited.require(value['schema']=='root_post_smollm3_sio12_complete_predecessors_v1'
            and value['reviewer']=='/root' and value['root_finalized']is True
            and value['complete_count']==len(value['rows'])==760
            and len(value['processes'])==146 and type(value['source_sha256'])is dict,
            'Exact completed catalog schema/counts')
        return raw
    if name in INSTALLATION_RECEIPTS:
        expected=INSTALLATION_RECEIPTS[name]
        inherited.require(wanted is None or wanted==expected,'Exact installed receipt pin')
        inherited.low.clear_window()
        raw=inherited.stable_bytes(path,64)
        raw.decode('utf8')
        inherited.require(hashlib.sha256(raw).hexdigest()==expected,'Installed receipt bytes changed')
        return raw
    if name in inherited.NON_UTF8:
        expected=inherited.NON_UTF8[name]
        inherited.require(wanted is None or wanted==expected,'Exact original Python encoding exception')
        return inherited.python_source_bytes(path,expected)
    if name in inherited.SPECIAL_TEXT_PINS and wanted is None:wanted=inherited.SPECIAL_TEXT_PINS[name]
    return inherited.text(path,wanted)
