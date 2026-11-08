"""No model/services: synthetic history fixtures exist only in test memory."""
import copy
from pathlib import Path
import sys
import subprocess
import types
import unittest
from unittest.mock import patch
ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
sys.path.insert(0,str(ROOT))
import minicpm5_frame1_mlp_runner_v1 as r


def catalog(rows=2,processes=1):
    return dict(reviewer='/root',root_finalized=True,all_planned_launches_accounted_for=True,
                runtime_physical_drain_required=True,complete_count=rows,
                rows=[dict(unit='test-'+str(i),original_outcome='failure'if i==0 else'never_launched')for i in range(rows)],
                processes=[dict(pid=100+i,start_ticks=str(i))for i in range(processes)],
                source_sha256={'/test/original.py':'1'*64},root_release_refs=[dict(path='/test/failed-release.json',sha256='2'*64)])


def history():
    anchor=catalog();smol=catalog(27,2);smol['schema']='root_post_smollm3_sio14_complete_predecessors_v1'
    smol_ref=dict(path=str(ROOT/'research/root-post-smol-sio14-complete-predecessors-20261108-v1.json'),sha256='3'*64)
    current=catalog(28,3);current['source_sha256'][smol_ref['path']]=smol_ref['sha256']
    current_ref=dict(path=str(ROOT/'research/nemotron9b-package-future-fixture-only.json'),sha256='4'*64)
    spec=dict(history_anchor=r.HISTORY_ANCHOR,required_sio14_catalog=smol_ref,current_catalog=current_ref,required_smol_predecessor_chain=['memory-only'])
    docs={r.HISTORY_ANCHOR['path']:anchor,smol_ref['path']:smol,current_ref['path']:current}
    return spec,docs


class RunnerControls(unittest.TestCase):
    def test_import_has_no_frameworks(self):
        self.assertFalse({'torch','torch_neuronx','transformers','neuronx_distributed'}&set(sys.modules))

    def test_import_from_actual_root_entrypoint_search_path(self):
        code="import sys;sys.path.insert(0,"+repr(str(ROOT))+");import minicpm5_frame1_mlp_runner_v1;assert not {'torch','torch_neuronx'}&set(sys.modules)"
        completed=subprocess.run([sys.executable,'-B','-c',code],cwd='/tmp',capture_output=True,text=True,timeout=15)
        self.assertEqual(completed.returncode,0,completed.stderr)

    def test_preserved_history_extension(self):
        a=catalog();c=catalog(3,2);self.assertIs(r.extend_catalog(c,a),c)

    def test_reject_history_omission_rewrite_and_changed_pin(self):
        for key in ('missing_row','rewrite_failure','missing_process','rewrite_process','source','release','duplicate','count','not_final','no_drain'):
            with self.subTest(key=key):
                a=catalog();c=catalog(3,2)
                if key=='missing_row':c['rows'].pop(0);c['complete_count']-=1
                if key=='rewrite_failure':c['rows'][0]['original_outcome']='success'
                if key=='missing_process':c['processes']=[]
                if key=='rewrite_process':c['processes'][0]['start_ticks']='999'
                if key=='source':c['source_sha256']['/test/original.py']='9'*64
                if key=='release':c['root_release_refs']=[]
                if key=='duplicate':c['rows'][-1]['unit']=c['rows'][0]['unit']
                if key=='count':c['complete_count']=99
                if key=='not_final':c['root_finalized']=False
                if key=='no_drain':c['runtime_physical_drain_required']=False
                with self.assertRaises(RuntimeError):r.extend_catalog(c,a)

    def test_requires_actual_smol_document(self):
        spec,docs=history();del docs[spec['required_sio14_catalog']['path']]
        with self.assertRaises(KeyError):r.current_history(spec,lambda ref:docs[ref['path']])

    def test_memory_fixture_chain_called_with_exact_anchor(self):
        spec,docs=history()
        with patch.object(r.smol_chain,'validate',return_value=dict(predecessor_anchor=r.HISTORY_ANCHOR))as validate:
            self.assertIs(r.current_history(spec,lambda ref:docs[ref['path']]),docs[spec['current_catalog']['path']])
            args=validate.call_args.args;self.assertEqual(args[0]['final_predecessor_catalog'],spec['required_sio14_catalog'])
            self.assertEqual(args[0]['required_smol_predecessor_chain'],spec['required_smol_predecessor_chain'])
            self.assertEqual(args[0]['source_sha256'][spec['required_sio14_catalog']['path']],'3'*64)

    def test_reject_missing_smol_units_and_external_owner(self):
        for kind in ('rows','processes'):
            spec,docs=history();smol=docs[spec['required_sio14_catalog']['path']];smol[kind].pop()
            if kind=='rows':smol['complete_count']-=1
            with self.assertRaises(RuntimeError):r.current_history(spec,lambda ref:docs[ref['path']])

    def test_current_history_cannot_omit_smol(self):
        spec,docs=history();spec['current_catalog']=r.HISTORY_ANCHOR
        with self.assertRaises(RuntimeError):r.current_history(spec,lambda ref:docs[ref['path']])

    def test_current_catalog_must_pin_smol(self):
        spec,docs=history();docs[spec['current_catalog']['path']]['source_sha256'].pop(spec['required_sio14_catalog']['path'])
        with self.assertRaises(RuntimeError):r.current_history(spec,lambda ref:docs[ref['path']])

    def test_terminal_chain_rejection_propagates(self):
        spec,docs=history()
        with patch.object(r.smol_chain,'validate',side_effect=RuntimeError('original terminal audit failed')):
            with self.assertRaisesRegex(RuntimeError,'original terminal audit failed'):r.current_history(spec,lambda ref:docs[ref['path']])

    def test_wrong_completed_chain_anchor_rejected(self):
        spec,docs=history()
        with patch.object(r.smol_chain,'validate',return_value=dict(predecessor_anchor={'path':'wrong'})):
            with self.assertRaises(RuntimeError):r.current_history(spec,lambda ref:docs[ref['path']])

    def test_strict_resource_contract(self):
        self.assertEqual(r.CAP,192*2**20)
        for phase in r.PHASES:
            props=r.props(phase)
            self.assertEqual(props['LimitFSIZE'],str(16*2**20));self.assertEqual(props['MemorySwapMax'],'0')
            self.assertEqual(props['RuntimeMaxSec'],'600'if phase=='replay'else'900')
            self.assertEqual(props['MemoryMax'],str((4 if phase=='replay'else 16)*2**30))
        self.assertEqual(r.props('replay')['AllowedCPUs'],'6')

    def test_serial_service_command_and_environment(self):
        spec=dict(units={p:'trn1-mini-f1-mlp-test-'+p for p in r.PHASES},output_root=str(ROOT/'minicpm5-frame1-mlp-r99-20261108-v1'),master_port=29571)
        ref=dict(path='/test/admission.json',sha256='a'*64)
        for phase in r.PHASES:
            command=r.service_command(spec,ref,'b'*32,phase)
            self.assertIn('--unit='+spec['units'][phase],command);self.assertIn('LimitFSIZE='+str(16*2**20),command)
            self.assertEqual(command[-10:],r.arguments('phase',ref,phase)[-10:])
        env=r.environment('cpu','c'*32,29571,1)
        self.assertEqual((env['RANK'],env['WORLD_SIZE'],env['MASTER_PORT']),('1','2','29571'))

    def test_exec_transition_accepts_only_exact_env_or_final_argv(self):
        spec=dict(units={p:'trn1-mini-f1-mlp-test-'+p for p in r.PHASES},output_root='/test/output',master_port=29571)
        ref=dict(path='/test/admission.json',sha256='a'*64);phase='native'
        group='0::'+r.p.GROUP_PREFIX+spec['units'][phase]+'.service'
        self.assertFalse(r.phase_exec_ready(None,spec,ref,'b'*32,phase))
        argv=r.arguments('phase',ref,phase)
        self.assertTrue(r.phase_exec_ready(dict(cgroup=group,argv=argv),spec,ref,'b'*32,phase))
        cmd=r.service_command(spec,ref,'b'*32,phase)
        self.assertFalse(r.phase_exec_ready(dict(cgroup=group,argv=cmd[cmd.index('/usr/bin/env'):]),spec,ref,'b'*32,phase))
        with self.assertRaises(RuntimeError):r.phase_exec_ready(dict(cgroup=group,argv=['unreviewed']),spec,ref,'b'*32,phase)
        with self.assertRaises(RuntimeError):r.phase_exec_ready(dict(cgroup='0::/wrong',argv=argv),spec,ref,'b'*32,phase)

    def test_historical_partition_does_not_read_raw_payload(self):
        raw='/test/original.pt';pins={'/test/source.py':'1'*64,raw:'2'*64}
        parts=dict(historical_payload_sha256={raw:'2'*64},provenance_sha256={'/test/source.py':'1'*64})
        with patch.object(r.history_partition,'partition',return_value=parts),patch.object(r.history_metadata,'sha',side_effect=lambda p,s:s)as sha:
            result=r.authenticate_history_sources(dict(history_metadata_aliases={}),dict(source_sha256=pins))
            self.assertEqual([call.args[0]for call in sha.call_args_list],['/test/source.py'])
            self.assertFalse(result['historical_raw_bytes_read'])

    def test_unaccounted_historical_payload_rejected(self):
        parts=dict(historical_payload_sha256={},provenance_sha256={})
        with patch.object(r.history_partition,'partition',return_value=parts):
            with self.assertRaises(RuntimeError):r.authenticate_history_sources(dict(history_metadata_aliases={}),dict(source_sha256={'/test/unknown.pt':'2'*64}))


if __name__=='__main__':unittest.main()
