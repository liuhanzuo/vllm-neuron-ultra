"""Read hash-bound completed Smol history and exact original wheel receipts.

No runtime discovery or admission: the caller must supply the completed-catalog
hash already bound by its independently reviewed specification. Existing ordinary
metadata limits and all subsequent predecessor/source checks remain in force.
"""
import hashlib,json,re
from pathlib import Path
import historical_b3_payload_partition_v1 as history
from minicpm5_layer0_runtime_auth_v1 import require
ROOT=Path(__file__).resolve().parent
CATALOG_LIMIT=16*2**20
RECEIPT_LIMIT=64
INSTALLATION_RECEIPTS={
 '/mnt/nvme/smollm3-cache-ablation-native-v3/target/vllm_neuron-0.2.2+lts.dist-info/REQUESTED':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 '/mnt/nvme/smollm3-cache-ablation-native-v3/target/vllm_neuron-0.2.2+lts.dist-info/INSTALLER':'ceebae7b8927a3227e5303cf5e0f1f7b34bb542ad7250ac03fbcde36ec2f1508',
}
def catalog_number(path):
 path=Path(path)
 if path.parent!=ROOT/'research':return None
 match=re.fullmatch(r'root-post-smol-sio([1-9][0-9]*)-complete-predecessors-([0-9]{8})-v1\.json',path.name)
 if match is None:return None
 number=int(match[1]);return number if 12<=number<=9999 else None

def accepts(path):return str(path)in INSTALLATION_RECEIPTS or catalog_number(path)is not None

def source_bytes(path,expected,stable_reader):
 path=Path(path)
 if str(path)in INSTALLATION_RECEIPTS:
  wanted=INSTALLATION_RECEIPTS[str(path)]
  require(expected is None or expected==wanted,'Exact original installed receipt hash')
  raw=stable_reader(path,RECEIPT_LIMIT);raw.decode('utf8')
  require(hashlib.sha256(raw).hexdigest()==wanted,'Original installed receipt changed')
  return raw
 number=catalog_number(path);require(number is not None,'Exact completed Smol catalog namespace')
 require(type(expected)is str and re.fullmatch('[0-9a-f]{64}',expected),'Reviewed completed-catalog digest required before reading')
 raw=stable_reader(path,CATALOG_LIMIT)
 require(hashlib.sha256(raw).hexdigest()==expected,'Completed Smol catalog fingerprint changed')
 value=json.loads(raw.decode('utf8'))
 require(type(value)is dict and value.get('schema')==f'root_post_smollm3_sio{number}_complete_predecessors_v1','Completed Smol catalog filename/schema binding')
 require(value.get('reviewer')=='/root'and value.get('root_finalized')is True and value.get('all_planned_launches_accounted_for')is True,'Actual finalized complete history required')
 rows=value.get('rows');processes=value.get('processes');pins=value.get('source_sha256')
 require(type(rows)is list and bool(rows)and type(processes)is list and type(pins)is dict and bool(pins),'Complete history shape')
 require(type(value.get('complete_count'))is int and value['complete_count']==len(rows),'Actual complete service count')
 require(all(type(row)is dict and type(row.get('unit'))is str for row in rows)and len({row['unit']for row in rows})==len(rows),'Unique complete service identities')
 require(all(type(row)is dict for row in processes),'Recorded process identity shape')
 history.pin_shape(pins)
 terminal=value.get('appended_terminal_audit')
 require(type(terminal)is dict and set(terminal)=={'path','sha256'}and type(terminal['path'])is str and type(terminal['sha256'])is str and re.fullmatch('[0-9a-f]{64}',terminal['sha256']),'Retained actual root terminal reference')
 require(pins.get(terminal['path'])==terminal['sha256'],'Completed terminal reference retained in source closure')
 return raw
