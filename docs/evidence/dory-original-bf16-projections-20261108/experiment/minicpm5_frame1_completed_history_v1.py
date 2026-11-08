"""Exact completed Mini pre-admission history; original v13 transport unchanged.

Only one retained, fixed-hash JSON catalog may use the existing32MiB envelope.
All other source/metadata classes keep the unmodified v13 policy.
"""
import hashlib
import json
from pathlib import Path
import nemotron9b_package_metadata_v13 as original
PATH=original.RESEARCH/'minicpm5-frame1-r1-pre-admission-complete-catalog-20261108-v1.json'
SHA='0d24c4eece03483119161b1566579afe63e659408bce329104c3d59438526e4d'
SIZE=10469465
NON_UTF8=original.NON_UTF8
python_source_bytes=original.python_source_bytes


def text(path,expected=None):
    path=Path(path)
    if path!=PATH:return original.text(path,expected)
    original.low.clear_window()
    original.require(expected in (None,SHA),'Exact completed pre-admission catalog fingerprint')
    raw=original.stable_bytes(path,original.ENVELOPE_LIMIT)
    original.require(len(raw)==SIZE and hashlib.sha256(raw).hexdigest()==SHA,'Exact complete historical catalog bytes')
    raw.decode('utf8');value=json.loads(raw)
    original.require(value['schema']=='root_post_minicpm5_frame1_pre_admission_attempt_v1' and
                     value['complete_count']==len(value['rows'])==919 and len(value['processes'])==170 and
                     value['root_finalized']is True and value['all_planned_launches_accounted_for']is True and
                     value['latest_producer_qualified']is False,'Actual failed pre-admission history, no success promotion')
    original.require(value['minicpm5_frame1_r1_pre_admission_append']['no_service_launched']is True and
                     all(row['never_launched']is True and row['owner']is None for row in value['rows'][-3:]),'Three original unlaunched service rows')
    return raw


def sha(path,expected=None):return hashlib.sha256(text(path,expected)).hexdigest()


def load(ref):
    if Path(ref['path'])!=PATH:return original.load(ref)
    original.require(type(ref)is dict and set(ref)in ({'path','sha256'},{'path','sha256','bytes'}),'Exact completed history reference')
    original.require(ref['sha256']==SHA and ('bytes'not in ref or type(ref['bytes'])is int and ref['bytes']==SIZE),'Exact completed history reference values')
    return json.loads(text(ref['path'],ref['sha256']))
