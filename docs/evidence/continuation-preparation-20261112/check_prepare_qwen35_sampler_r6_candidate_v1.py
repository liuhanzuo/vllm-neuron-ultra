"""Frozen r5 117-row metadata fixture plus synthetic future metadata only."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import prepare_qwen35_sampler_r6_candidate_v1 as p
import qwen35_sampler_runtime_contract_v6 as contract


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixed={}
        for ref in p.FIXED.values():
            data=Path(ref['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==ref['sha256']
            cls.fixed[ref['path']]=json.loads(data)

    def fixture(self, *, append=True):
        values=copy.deepcopy(self.fixed)
        spec=values[p.FIXED['spec']['path']];overlay=values[p.FIXED['overlay']['path']]
        identities={path:copy.deepcopy(row['identity'])for block in (spec['input_extension'],overlay['input_extension'])for path,row in block.items()}
        future=dict(terminal={'path':str(Q/'continuation-handoff-synthetic-idle.json'),'sha256':'a'*64},
            prerequisite_exit={'path':str(Q/'synthetic-workload-actual-exit.json'),'sha256':'b'*64},
            catalog={'path':str(Q/'synthetic-complete-catalog.json'),'sha256':'c'*64},run_name='r6-synthetic',
            output_root='/mnt/nvme/neuron-work/qwen35-sampler-pair-r6-synthetic',
            overlay_output_root='/mnt/nvme/minicpm5-layer1-operator-native-qwen-r6-synthetic',
            running_handoff_path=str(Q/'continuation-handoff-synthetic-qwen-r6-running.json'),
            overlay_path=str(Q/'qwen35-synthetic-r6-history-overlay.json'),spec_path=str(Q/'qwen35-synthetic-r6-spec.json'),
            review_candidate_path=str(Q/'qwen35-synthetic-r6-review-candidate.json'))
        cat=copy.deepcopy(values[p.FIXED['catalog']['path']]);cat['current_handoff']=future['terminal']
        for row in cat['runs']:values[row['terminal']['path']]={'synthetic_saved_terminal_for_fixture':True}
        new=dict(terminal=copy.deepcopy(future['prerequisite_exit']),tool={'kind':'tool_session','session_id':1,'terminal_chunk':'abc123','exit_code':1},
                 owners=[{'pid':123,'start_ticks':'456'}],units=['trn1-synthetic-completed-workload'],
                 source_sha256={str(Q/'synthetic-future-source.py'):'d'*64})
        if append:cat['runs'].append(new)
        values[future['catalog']['path']]=cat
        values[future['terminal']['path']]={'serial_execution':True,'live_tool_sessions':[],'active_heavy_units':[]}
        values[future['prerequisite_exit']['path']]={'synthetic_actual_tool':copy.deepcopy(new['tool'])}
        read_calls=[]
        def read(ref):
            read_calls.append(copy.deepcopy(ref))
            identity=copy.deepcopy(identities.get(ref['path'],dict(realpath=ref['path'],device=1,inode=2,size=1,mtime_ns=3,ctime_ns=3)))
            return dict(value=copy.deepcopy(values[ref['path']]),identity=identity)
        return future,values,identities,read,read_calls

    def run_fixture(self,mutate=None):
        f=self.fixture()
        if mutate:mutate(*f)
        return p.prepare(f[0],f[3],Mock()),f

    def test01_preserves_all47929_sources_117rows_and_both_input_sets(self):
        result,f=self.run_fixture();future,values,_,_,_=f
        old=values[p.FIXED['spec']['path']];old_overlay=values[p.FIXED['overlay']['path']]
        spec=result['artifacts'][future['spec_path']];overlay=result['artifacts'][future['overlay_path']]
        self.assertTrue(all(spec['source_sha256'].get(path)==sha for path,sha in old['source_sha256'].items()))
        self.assertTrue(all(spec['input_extension'].get(path)==row for path,row in old['input_extension'].items()))
        self.assertTrue(all(overlay['input_extension'].get(path)==row for path,row in old_overlay['input_extension'].items()))
        self.assertEqual(len(spec['source_sha256']),47936)
        self.assertEqual(result['evidence']['candidate_serial_rows'],118)
        for key in ('limits','native_python','lock_identity','frozen_science_delivery','automatic_successor'):
            self.assertEqual(spec[key],old[key])
        self.assertEqual(spec['source_sha256'][p.LEAF['path']],p.LEAF['sha256'])
        recovered={path:sha for path,sha in old['source_sha256'].items()if path not in old_overlay['source_extension']}
        self.assertFalse(set(recovered)&set(overlay['source_extension']))
        self.assertEqual({**recovered,**overlay['source_extension']},spec['source_sha256'])

    def test02_candidate_review_fails_original_runtime_contract(self):
        result,f=self.run_fixture();future=f[0]
        spec=result['artifacts'][future['spec_path']];review=result['artifacts'][future['review_candidate_path']]
        self.assertFalse(review['passed']);self.assertFalse(review['launch_authorized'])
        with self.assertRaisesRegex(ValueError,'root review/admission'):
            contract.validate_spec(spec,review,result['references']['spec_path'])
        self.assertFalse(result['evidence']['full_recovered_witness_closure_verified'])
        self.assertFalse(result['evidence']['output_freshness_observed'])

    def test03_changed_or_missing_old_serial_row_rejected(self):
        for kind in ('missing','code','owner','source'):
            f=self.fixture();rows=f[1][f[0]['catalog']['path']]['runs']
            if kind=='missing':rows.pop(5)
            if kind=='code':rows[5]['tool']['exit_code']=123
            if kind=='owner':rows[5]['owners']=[]
            if kind=='source':rows[5]['source_sha256']={}
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'117 exact original'):
                p.prepare(f[0],f[3],Mock())

    def test04_new_row_conflict_with_old_spec_pin_rejected(self):
        f=self.fixture();old=f[1][p.FIXED['spec']['path']];path=next(iter(old['source_sha256']))
        f[1][f[0]['catalog']['path']]['runs'][-1]['source_sha256'][path]='0'*64
        with self.assertRaises(ValueError):p.prepare(f[0],f[3],Mock())

    def test05_conflicting_pins_between_new_catalog_rows_rejected(self):
        f=self.fixture();rows=f[1][f[0]['catalog']['path']]['runs'];extra=copy.deepcopy(rows[-1])
        extra['terminal']={'path':str(Q/'synthetic-other-exit.json'),'sha256':'e'*64}
        extra['source_sha256']={str(Q/'synthetic-future-source.py'):'f'*64};rows.append(extra)
        f[1][extra['terminal']['path']]={'synthetic':True}
        with self.assertRaisesRegex(ValueError,'across complete future catalog'):p.prepare(f[0],f[3],Mock())

    def test06_old_input_identity_is_not_refreshed(self):
        f=self.fixture();path=p.FIXED['catalog']['path']
        self.assertIn(path,f[2]);f[2][path]['mtime_ns']+=1
        with self.assertRaisesRegex(ValueError,'Old input identity'):p.prepare(f[0],f[3],Mock())

    def test07_live_future_terminal_and_missing_prerequisite_rejected(self):
        for kind in ('live','missing'):
            f=self.fixture(append=kind!='missing')
            if kind=='live':f[1][f[0]['terminal']['path']]['live_tool_sessions']=[39484]
            with self.subTest(kind=kind),self.assertRaises(ValueError):p.prepare(f[0],f[3],Mock())

    def test08_all_future_bindings_required_and_historical_outputs_protected(self):
        f=self.fixture()
        for key in f[0]:
            future=copy.deepcopy(f[0]);future.pop(key)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'must be explicit'):
                p.prepare(future,f[3],Mock())
        future=copy.deepcopy(f[0]);future['spec_path']=p.FIXED['spec']['path']
        with self.assertRaisesRegex(ValueError,'cannot overwrite'):p.prepare(future,f[3],Mock())

    def test09_source_review_or_leaf_replacement_rejected(self):
        for kind in ('runtime','leaf'):
            f=self.fixture()
            if kind=='runtime':f[1][p.FIXED['runtime_review']['path']]['runtime_admitted']=True
            else:f[1][p.FIXED['leaf_review']['path']]['source']['sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):p.prepare(f[0],f[3],Mock())

    def test10_candidate_refs_bind_exact_encoding_and_no_mutation(self):
        f=self.fixture();before=copy.deepcopy(f[1]);result=p.prepare(f[0],f[3],Mock())
        self.assertEqual(f[1],before)
        for key,ref in result['references'].items():
            self.assertEqual(hashlib.sha256(p.encode(result['artifacts'][ref['path']])).hexdigest(),ref['sha256'])

    def test11_real_JSON_reader_has_stable_hash_and_does_not_refresh_identity(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'fixture.json';raw=b'{"fixture":true}\n';path.write_bytes(raw)
            ref={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()};reader=p.JsonMetadata(Mock())
            value=reader.read(ref);self.assertEqual(value['value'],{'fixture':True})
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'cannot be refreshed'):reader.read(ref)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'identity and hash'):reader.read(ref)

    def test12_guard_fails_before_first_read(self):
        reader=Mock();guard=Mock(side_effect=RuntimeError('synthetic denied'))
        with self.assertRaisesRegex(RuntimeError,'synthetic denied'):p.prepare({},reader,guard)
        reader.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
