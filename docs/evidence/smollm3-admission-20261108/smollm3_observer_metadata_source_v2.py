"""Exact inherited metadata/encoding exceptions; never reads model payloads."""
from pathlib import Path
import hashlib
import nemotron9b_package_metadata_v7 as inherited

# Exact original wheel installation receipts: authenticated by retained wheel.json.
# No general extensionless-file exception and no model/native payload reads.
INSTALLATION_RECEIPTS = {
    '/mnt/nvme/smollm3-cache-ablation-native-v3/target/vllm_neuron-0.2.2+lts.dist-info/REQUESTED':
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    '/mnt/nvme/smollm3-cache-ablation-native-v3/target/vllm_neuron-0.2.2+lts.dist-info/INSTALLER':
        'ceebae7b8927a3227e5303cf5e0f1f7b34bb542ad7250ac03fbcde36ec2f1508',
}

# The inherited reader enumerates named/schema-checked 32MiB Nemotron envelopes.
# Everything else keeps its ordinary8MiB cap and text type restrictions.
def source_bytes(path,wanted=None):
    path=Path(path);name=str(path)
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
