"""Pure synthetic strict-pair validation tests; no physical/native/raw work."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research import minicpm5_sdk_physical_evidence_v3 as e

def props(unit,legacy=False):
    p=dict(Id=unit+'.service',MainPID='0',ControlPID='0',ActiveState='inactive',SubState='dead',
           LoadState='not-found',InvocationID='',ExecMainPID='0',ExecMainCode='0',ExecMainStatus='0',Result='success')
    if not legacy:p['Job']=''
    return p

def group(unit,kind=0):
    path=e.GROUP_PREFIX+unit+'.service'
    return [dict(group=path,absent=True),
            dict(group=path,absent=False,cgroups=2),
            dict(group=path,absent=True,pids=[],directories=[]),
            dict(group=path,absent=False,pids=[],directories=['/sys/fs/cgroup'+path,'/sys/fs/cgroup'+path+'/child'])][kind]

def row(unit,legacy=False,kind=0):
    sample=dict(properties=props(unit,legacy),group=group(unit,kind))
    return dict(unit=unit,samples=[copy.deepcopy(sample),copy.deepcopy(sample)])

def tree(rows):
    return dict(complete=True,physical_checks_performed=True,all_current_and_prior_services_preserved=True,
        all_additional_owners_checked=[dict(pid=8,start_ticks='9')],lock_identity=dict(inode=5),
        nemotron=dict(complete=True,fresh_physical_drain_verified=True,services=1308,observations=rows),
        full_model_qualified=False,new_speedup_claim=False)

def observation(t,supplemental=None):
    return dict(tree=t,supplemental_observations=[] if supplemental is None else supplemental)

def pair(value):
    return dict(schema=e.PAIR_SCHEMA,first=copy.deepcopy(value),second=copy.deepcopy(value))


class Strict(unittest.TestCase):
    def setUp(self):
        self.names=['trn1-test-%d'%i for i in range(1308)]
        self.t=tree([row(name,kind=i%4)for i,name in enumerate(self.names)])
        self.value=observation(self.t)

    def test_full1308_and_all_four_real_group_schemas(self):
        self.assertEqual(e.validate_pair(pair(self.value),self.names,self.t['all_additional_owners_checked'],self.t['lock_identity']),set(self.names))

    def test_legacy_job_missing_requires_exact_supplement_and_original_unchanged(self):
        self.t['nemotron']['observations'][0]=row(self.names[0],legacy=True)
        original=copy.deepcopy(self.t)
        self.assertEqual(e.inspect_tree(self.t)['needs_supplement'],{self.names[0]})
        with self.assertRaises(RuntimeError):e.validate_pair(pair(self.value))
        self.value['supplemental_observations']=[dict(unit=self.names[0],properties=props(self.names[0]),group=group(self.names[0]))]
        self.assertEqual(e.validate_pair(pair(self.value)),set(self.names))
        self.assertEqual(self.t,original)
        self.assertNotIn('Job',self.t['nemotron']['observations'][0]['samples'][0]['properties'])

    def test_group_only_extra_requires_supplement_preserves_extra_set(self):
        extra='trn1-extra'
        self.t['native_group']=group(extra)
        self.assertEqual(e.inspect_tree(self.t)['needs_supplement'],{extra})
        with self.assertRaises(RuntimeError):e.validate_pair(pair(self.value))
        self.value['supplemental_observations']=[dict(unit=extra,properties=props(extra),group=group(extra))]
        self.assertEqual(e.validate_pair(pair(self.value),self.names),set(self.names)|{extra})

    def test_real_stage_single_observation_covered_by_two_full_trees(self):
        r=self.t['nemotron']['observations'][0]
        self.t['nemotron']['observations'][0]=dict(stage='controller',**r['samples'][0])
        self.assertEqual(e.validate_pair(pair(self.value)),set(self.names))

    def test_groups_list_detailed_form(self):
        for sample in self.t['nemotron']['observations'][0]['samples']:
            sample['groups']=[sample.pop('group')]
        self.assertEqual(e.validate_pair(pair(self.value)),set(self.names))

    def test_missing_all_samples_fails(self):
        self.t['nemotron']['observations']=[{'unit':name}for name in self.names]
        with self.assertRaises(RuntimeError):e.validate_pair(pair(self.value),self.names)

    def test_missing_one_full_sample_set_fails(self):
        self.t['nemotron']['observations'][0].pop('samples')
        with self.assertRaises(RuntimeError):e.validate_pair(pair(self.value))

    def test_mismatched_existing_id_fails_even_when_set_unchanged(self):
        for sample in self.t['nemotron']['observations'][0]['samples']:sample['properties']['Id']=self.names[1]+'.service'
        with self.assertRaises(RuntimeError):e.validate_pair(pair(self.value))

    def test_nonempty_group_fails(self):
        for sample in self.t['nemotron']['observations'][0]['samples']:
            sample['group']=dict(group=e.GROUP_PREFIX+self.names[0]+'.service',absent=False,pids=[777],directories=[])
        with self.assertRaises(RuntimeError):e.validate_pair(pair(self.value))

    def test_pending_job_fails(self):
        for sample in self.t['nemotron']['observations'][0]['samples']:sample['properties']['Job']='1234'
        with self.assertRaises(RuntimeError):e.validate_pair(pair(self.value))

    def test_wrong_sample_count_or_difference_fails(self):
        original=copy.deepcopy(self.value)
        for samples in [[],[row(self.names[0])['samples'][0]],row(self.names[0])['samples']*2]:
            self.value=copy.deepcopy(original);self.value['tree']['nemotron']['observations'][0]['samples']=samples
            with self.subTest(size=len(samples)),self.assertRaises(RuntimeError):e.validate_pair(pair(self.value))
        self.value=copy.deepcopy(original);self.value['tree']['nemotron']['observations'][0]['samples'][1]['properties']['Result']='different'
        with self.assertRaises(RuntimeError):e.validate_pair(pair(self.value))

    def test_current_complete_tree_difference_rejected(self):
        p=pair(self.value);p['second']['tree']['opaque_actual_field']='different'
        with self.assertRaises(RuntimeError):e.validate_pair(p)

    def test_owner_and_lock_binding(self):
        for kw in [dict(expected_owners=[]),dict(lock={'inode':9})]:
            with self.subTest(kw=kw),self.assertRaises(RuntimeError):e.validate_pair(pair(self.value),**kw)

    def test_required_unit_substitution_and_count_rejected(self):
        for required in [self.names[:-1],self.names[:-1]+['trn1-other']]:
            with self.assertRaises(RuntimeError):e.validate_pair(pair(self.value),required)

    def test_illegal_terminal_combinations_and_live_pid(self):
        for key,value in [('ActiveState','activating'),('SubState','running'),('MainPID','777'),('ControlPID','777'),('Job',None),('Job',False)]:
            p=pair(self.value)
            for branch in ['first','second']:
                for sample in p[branch]['tree']['nemotron']['observations'][0]['samples']:sample['properties'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(RuntimeError):e.validate_pair(p)

    def test_missing_job_nonlegacy_schema_rejected(self):
        for sample in self.t['nemotron']['observations'][0]['samples']:
            sample['properties'].pop('Job');sample['properties']['unknown']='field'
        with self.assertRaises(RuntimeError):e.inspect_tree(self.t)

    def test_group_wrong_unit_path_or_escape_rejected(self):
        for path in [e.GROUP_PREFIX+self.names[1]+'.service',e.GROUP_PREFIX+self.names[0]+'.service/child',
                     '/tmp/'+self.names[0]+'.service',e.GROUP_PREFIX+'../bad.service']:
            p=pair(self.value)
            for branch in ['first','second']:
                for sample in p[branch]['tree']['nemotron']['observations'][0]['samples']:sample['group']['group']=path
            with self.subTest(path=path),self.assertRaises(RuntimeError):e.validate_pair(p)

    def test_group_summary_count_absent_and_extra_claims(self):
        for bad in [dict(group=e.GROUP_PREFIX+self.names[0]+'.service',absent=False),
                    dict(group=e.GROUP_PREFIX+self.names[0]+'.service',absent=False,cgroups=0),
                    dict(group=e.GROUP_PREFIX+self.names[0]+'.service',absent=False,cgroups=True),
                    dict(group=e.GROUP_PREFIX+self.names[0]+'.service',absent=False,cgroups=257),
                    dict(group=e.GROUP_PREFIX+self.names[0]+'.service',absent=True,cgroups=1),
                    dict(group=e.GROUP_PREFIX+self.names[0]+'.service',absent=True,populated=True),
                    dict(group=e.GROUP_PREFIX+self.names[0]+'.service')]:
            with self.subTest(bad=bad),self.assertRaises(RuntimeError):e.empty_group(bad)

    def test_detailed_group_directories_root_escape_or_pid_rejected(self):
        original=group(self.names[0],3)
        for field,value in [('pids',[7]),('directories',[]),('directories',['/sys/fs/cgroup/outside']),
                            ('directories',[original['directories'][0]+'/child']),
                            ('directories',[original['directories'][0],original['directories'][0]+'/../bad'])]:
            bad=copy.deepcopy(original);bad[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(RuntimeError):e.empty_group(bad)

    def test_group_and_groups_cannot_conflict_or_duplicate(self):
        sample=self.t['nemotron']['observations'][0]['samples'][0]
        sample['groups']=[sample['group']]
        with self.assertRaises(RuntimeError):e.observation(sample,self.names[0])
        sample.pop('group');sample['groups']*=2
        with self.assertRaises(RuntimeError):e.observation(sample,self.names[0])

    def test_supplement_omission_extra_duplicate_or_mismatch(self):
        self.t['nemotron']['observations'][0]=row(self.names[0],legacy=True)
        good=dict(unit=self.names[0],properties=props(self.names[0]),group=group(self.names[0]))
        for rows in [[],[good,copy.deepcopy(good)],[dict(unit=self.names[1],properties=props(self.names[1]),group=group(self.names[1]))]]:
            with self.subTest(rows=len(rows)),self.assertRaises(RuntimeError):e.validate_observation(observation(self.t,rows))
        bad=copy.deepcopy(good);bad['properties']['Job']='1'
        with self.assertRaises(RuntimeError):e.validate_observation(observation(self.t,[bad]))

    def test_supplement_state_must_agree_original(self):
        self.t['nemotron']['observations'][0]=row(self.names[0],legacy=True)
        p=props(self.names[0]);p.update(LoadState='loaded',InvocationID='changed',ActiveState='failed',SubState='failed')
        with self.assertRaises(RuntimeError):e.validate_observation(observation(self.t,[dict(unit=self.names[0],properties=p,group=group(self.names[0]))]))

    def test_known_units_limit_no_arbitrary_group(self):
        self.t['model_group']=group('trn1-unknown')
        with self.assertRaises(RuntimeError):e.inspect_tree(self.t,self.names)

    def test_collect_pair_two_complete_calls_and_exact_missing_callbacks(self):
        self.t['nemotron']['observations'][0]=row(self.names[0],legacy=True)
        self.t['model_group']=group('trn1-extra')
        done=dict(nemotron=dict(history=dict(rows=[dict(unit=x)for x in self.names])),
                  owners=self.t['all_additional_owners_checked'],lock_identity=self.t['lock_identity'])
        guard=Mock();physical=Mock(side_effect=lambda *args:copy.deepcopy(self.t))
        get_props=Mock(side_effect=props);get_group=Mock(side_effect=lambda path:group(e.group_unit(path)))
        result=e.collect_pair(done,guard,physical,get_props,get_group,known_units=set(self.names)|{'trn1-extra'})
        self.assertEqual(physical.call_count,2)
        self.assertEqual([x.args[0]for x in get_props.call_args_list],['trn1-extra',self.names[0]]*2)
        self.assertEqual(e.validate_pair(result,self.names,done['owners'],done['lock_identity']),set(self.names)|{'trn1-extra'})
        self.assertNotIn('Job',self.t['nemotron']['observations'][0]['samples'][0]['properties'])

    def test_collect_unknown_group_rejected_before_any_supplement_callback(self):
        self.t['model_group']=group('trn1-not-in-authenticated-saved')
        done=dict(nemotron=dict(history=dict(rows=[dict(unit=x)for x in self.names])),
                  owners=self.t['all_additional_owners_checked'],lock_identity=self.t['lock_identity'])
        read=Mock()
        with self.assertRaises(RuntimeError):e.collect_pair(done,lambda:None,lambda *args:self.t,read,read,known_units=self.names)
        read.assert_not_called()

    def test_collect_rejects_invalid_tree_before_supplement_reads(self):
        self.t['nemotron']['observations'][0].pop('samples')
        done=dict(nemotron=dict(history=dict(rows=[dict(unit=x)for x in self.names])),
                  owners=self.t['all_additional_owners_checked'],lock_identity=self.t['lock_identity'])
        get=Mock()
        with self.assertRaises(RuntimeError):e.collect_pair(done,lambda:None,lambda *args:self.t,get,get)
        get.assert_not_called()

    def test_metadata_only_module_imports(self):
        self.assertFalse({'torch','numpy','neuronxcc','torch_neuronx'}&set(sys.modules))


class OwnerCompleteness(unittest.TestCase):
    def owner(self, name, pid=12, tagged=True):
        value=dict(pid=pid,start_ticks='123',ppid=8,argv=['python3','-B','observer.py'],
                   cgroup='0::'+e.GROUP_PREFIX+name+'.service')
        if tagged:value.update(unit=name,invocation_id='a'*32)
        return value

    def fixture(self):
        names=['trn1-synthetic-%d'%i for i in range(1308)]
        t=tree([row(name,legacy=True) for name in names])
        owners=[self.owner('trn1-owner-only'),self.owner('trn1-owner-only',tagged=False),
                self.owner('trn1-cgroup-only',pid=14,tagged=False)]
        t['all_additional_owners_checked']=owners
        done=dict(nemotron=dict(history=dict(rows=[dict(unit=x)for x in names])),
                  owners=owners,lock_identity=t['lock_identity'])
        return names,t,done

    def test_complete_owner_only_scope_requires_real_double_supplement(self):
        names,t,done=self.fixture();before=copy.deepcopy(t)
        known=set(names)|{'trn1-owner-only','trn1-cgroup-only'}
        physical=Mock(side_effect=lambda *args:copy.deepcopy(t));get=Mock(side_effect=props)
        get_group=Mock(side_effect=lambda path:group(e.group_unit(path)))
        out=e.collect_pair(done,lambda:None,physical,get,get_group,known_units=known)
        self.assertEqual(physical.call_count,2)
        self.assertEqual([c.args[0] for c in get.call_args_list],sorted(known)*2)
        self.assertEqual(get_group.call_count,2*len(known))
        self.assertEqual(e.validate_pair(out,names,done['owners'],done['lock_identity'],known_units=known),known)
        self.assertEqual(t,before)

    def test_owner_only_evidence_is_never_sufficient_by_itself(self):
        _,t,_=self.fixture()
        with self.assertRaises(RuntimeError):e.validate_pair(pair(observation(t)))
        scan=e.inspect_tree(t)
        good=[dict(unit=n,properties=props(n),group=group(n)) for n in sorted(scan['units'])]
        for name in ('trn1-owner-only','trn1-cgroup-only'):
            bad=[x for x in good if x['unit']!=name]
            with self.subTest(name=name),self.assertRaises(RuntimeError):e.validate_pair(pair(observation(t,bad)))

    def test_owner_shape_mismatch_or_incomplete_identity_rejected(self):
        _,t,_=self.fixture()
        for field,value in [('unit','trn1-substituted'),('pid',True),('ppid',0),('start_ticks',''),
                            ('argv',[]),('invocation_id','bad'),('cgroup','0::'+e.GROUP_PREFIX+'../escape.service')]:
            bad=copy.deepcopy(t);bad['all_additional_owners_checked'][0][field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):e.inspect_tree(bad)
        bad=copy.deepcopy(t);bad['all_additional_owners_checked'][1]['argv']=['different']
        with self.assertRaises(RuntimeError):e.inspect_tree(bad)

    def test_exact_known_set_rejects_missing_and_extra_before_callbacks(self):
        names,t,done=self.fixture();known=set(names)|{'trn1-owner-only','trn1-cgroup-only'}
        for bad in (known-{'trn1-owner-only'},known|{'trn1-missing'}):
            get=Mock()
            with self.assertRaises(RuntimeError):e.collect_pair(done,lambda:None,lambda *a:t,get,get,known_units=bad)
            get.assert_not_called()

    def test_loaded_owner_only_requires_original_invocation_and_main_identity(self):
        _,t,_=self.fixture();scan=e.inspect_tree(t)
        supplements=[dict(unit=n,properties=props(n),group=group(n))for n in sorted(scan['units'])]
        target=next(x for x in supplements if x['unit']=='trn1-owner-only')
        target['properties'].update(LoadState='loaded',InvocationID='a'*32,ExecMainPID='12')
        e.validate_pair(pair(observation(t,supplements)))
        for key,value in [('InvocationID','b'*32),('ExecMainPID','13')]:
            bad=copy.deepcopy(supplements);next(x for x in bad if x['unit']=='trn1-owner-only')['properties'][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):e.validate_pair(pair(observation(t,bad)))

    def test_unanchored_owner_only_loaded_retains_unknown_old_invocation(self):
        _,t,_=self.fixture();scan=e.inspect_tree(t)
        supplements=[dict(unit=n,properties=props(n),group=group(n))for n in sorted(scan['units'])]
        target=next(x for x in supplements if x['unit']=='trn1-cgroup-only')
        target['properties'].update(LoadState='loaded',InvocationID='b'*32,ExecMainPID='14')
        proof=pair(observation(t,supplements))
        e.validate_pair(proof)
        bad=copy.deepcopy(proof)
        next(x for x in bad['second']['supplemental_observations'] if x['unit']=='trn1-cgroup-only')['properties']['InvocationID']='c'*32
        with self.assertRaises(RuntimeError):e.validate_pair(bad)
        for key,value in [('MainPID','14'),('ControlPID','14'),('Job','123')]:
            bad=copy.deepcopy(supplements)
            next(x for x in bad if x['unit']=='trn1-cgroup-only')['properties'][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):e.validate_pair(pair(observation(t,bad)))

    def test_full_explicit_supplement_fields_and_stability(self):
        _,t,_=self.fixture();scan=e.inspect_tree(t)
        supplements=[dict(unit=n,properties=props(n),group=group(n))for n in sorted(scan['units'])]
        for key in ('Job','InvocationID','LoadState','ExecMainPID','ExecMainStatus','Result'):
            bad=copy.deepcopy(supplements);bad[0]['properties'].pop(key)
            with self.subTest(key=key),self.assertRaises(RuntimeError):e.validate_pair(pair(observation(t,bad)))
        bad=pair(observation(t,supplements));bad['second']['supplemental_observations'][0]['properties']['Result']='changed'
        with self.assertRaises(RuntimeError):e.validate_pair(bad)

    def test_actual_saved_metadata_1311_to_1316_preserves_legacy1313(self):
        import hashlib
        from research import minicpm5_sdk_physical_evidence_v2 as old
        from research.audit_minicpm5_layer1_sdk_batch_layout_result_v5 import legacy_drain_units
        base=Path(__file__).resolve().parent
        raw=(base/'minicpm5-layer1-sdk-batch-layout-r2-actual-exit-20261111-v1.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),'aa4c60e6eff2b0800eb470f1d3dbdb8180abfc1727da9c3393b861c6f0b26a43')
        ref=json.loads(raw)['outputs']['drains'];raw=Path(ref['path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['sha256'])
        t=json.loads(raw)['latest_physical'];before=copy.deepcopy(t)
        prior=old.inspect_tree(t);legacy=legacy_drain_units(t);current=e.inspect_tree(t)
        self.assertEqual((len(prior['units']),len(legacy),len(current['units']),len(current['needs_supplement'])),(1311,1313,1316,1316))
        self.assertLessEqual(prior['units'],legacy);self.assertLessEqual(legacy,current['units'])
        self.assertEqual(current['units']-legacy,{'trn1-mini-layer1-own-history-r1-20261110',
            'trn1-mini-layer1-cross-history-r1-20261110','trn1-mini-layer1-kv-factorial-r1-20261110'})
        self.assertEqual(len(t['all_additional_owners_checked']),1642)
        self.assertEqual(len(current['owner_only']),5)
        # Pure synthetic callbacks over authenticated metadata; never physical.
        def synthetic_props(name):
            previous=current['properties'].get(name)
            return dict(previous[0],Job='') if previous else props(name)
        required=[]
        # Actual historical unit set is exactly the original observed properties.
        required=sorted(current['properties'])
        self.assertEqual(len(required),1308)
        done=dict(nemotron=dict(history=dict(rows=[dict(unit=n)for n in required])),
            owners=t['all_additional_owners_checked'],lock_identity=t['lock_identity'])
        physical=Mock(side_effect=lambda *a:copy.deepcopy(t));get=Mock(side_effect=synthetic_props)
        group_get=Mock(side_effect=lambda path:group(e.group_unit(path)))
        proof=e.collect_pair(done,lambda:None,physical,get,group_get,known_units=current['units'])
        self.assertEqual(get.call_count,2632);self.assertEqual(group_get.call_count,2632)
        self.assertEqual(physical.call_count,2)
        self.assertEqual(e.validate_pair(proof,required,done['owners'],done['lock_identity'],known_units=current['units']),current['units'])
        self.assertEqual(t,before)

if __name__=='__main__':unittest.main(verbosity=2)
