"""Original88 SDK-only companion; no native imports, arithmetic, or admission."""
from contextlib import contextmanager
import copy
import functools
import types

import minicpm5_layer0_capture_observer_v2 as base
import minicpm5_python_trace_observer_v2 as native
from research import minicpm5_python_scale_trace_candidate_20261107_v3 as trace

require=base.require
SCHEMA=native.SCHEMA
SnapshotSink=base.SnapshotSink
SOURCE_SHA='c436c003161c9fd8d8d8a2a258284026b727aabca64d596c845d54ca776de5e8'
SOURCE_PATH='/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py'


def sdk_descriptor(binding):
    require(binding['route']in('sdk_prefill','sdk_decode'),'Exact SDK trace route')
    points=list(binding['events'].values())
    require([x['role']for x in points]==(['qk0']if binding['route']=='sdk_prefill'else['qk0','qk1']),
            'Original ordered SDK scale points')
    return dict(route=binding['route'],source=copy.deepcopy(binding['source']),function=binding['function'],
                line_metadata=copy.deepcopy(binding['line_metadata']),points=copy.deepcopy(points))


def validate_sdk_operation(value,expected,role,bridged=True):
    require(type(value)is dict and set(value)==native.BASE_KEYS|({'schema','operand_binding_basis'}if bridged else set()),
            'Exact SDK Python scale provenance fields')
    require(value['observation']=='actual_python_local_after_assignment_before_next_statement'and
            value['scaled_value_observed']is True and value['scaled_value_recomputed']is False and
            value['aten_operator_observed']is False,'Original assigned local only')
    require(value['operator']=='python.divide'and value['operator_identity_basis']=='pinned_source_AST'and
            type(value['scalar'])in(int,float)and value['scalar']==128**.5 and
            value['scalar_origin']=='derived_from_pinned_expression_and_observed_head_dim','Original SDK division provenance')
    require(expected['route']in('sdk_prefill','sdk_decode')and value['route']==expected['route'],'Exact SDK route')
    points=[x for x in expected['points']if x['role']==role];require(len(points)==1,'Unique expected SDK role')
    point=points[0]
    require(value['source']==expected['source']and value['function']==expected['function']and
            value['line_metadata']==expected['line_metadata']and value['assignment_line']==point['assignment_line']and
            value['event_line']==point['event_line']and value['local_name']==point['variable'],'Exact source/code/line/assigned local')
    if bridged:require(value['schema']==SCHEMA and value['operand_binding_basis']==native.OPERAND_BASIS,'Explicit actual predecessor binding')
    return True


class SDKCapture(base.Capture):
    def bind_trace(self,expected):
        require(self.arm in('C_Hd_control','C_Hc_history_generation')and not hasattr(self,'trace_expected'),'One SDK trace binding')
        require(expected['route']==('sdk_prefill'if self.phase=='prefill'else'sdk_decode')and self.route==expected['route'],
                'Trace inside original phase route wrapper')
        self.trace_expected=copy.deepcopy(expected);self.scale_provenance=[]
    def observe_scale(self,*args,**kwargs):
        raise RuntimeError('Aten dispatch forbidden in SDK Python trace')
    def scaled_local(self,role,actual,provenance):
        require(hasattr(self,'trace_expected'),'Authenticated SDK trace binding required')
        validate_sdk_operation(provenance,self.trace_expected,role,bridged=False)
        trace.capture_bridge(self,role,actual,provenance)
    def emit(self,label,value,selection=None):
        if label.endswith('_scale_operation'):
            require(hasattr(self,'trace_expected'),'Scale evidence before trace binding')
            roles=[point['role']for point in self.trace_expected['points']]
            index=len(self.scale_provenance)
            require(index<len(roles)and label==roles[index]+'_scale_operation','Exact ordered one-per-role SDK scale events')
            value=dict(value,schema=SCHEMA);validate_sdk_operation(value,self.trace_expected,roles[index])
            self.scale_provenance.append(copy.deepcopy(value))
        return super().emit(label,value,selection)
    def complete(self):
        value=super().complete()
        require(hasattr(self,'trace_expected'),'Missing SDK Python capture')
        value.update(scale_observation_schema=SCHEMA,scale_provenance=copy.deepcopy(self.scale_provenance),
                     trace_contract=copy.deepcopy(self.trace_expected),aten_dispatch_used=False)
        validate_sdk_capture_summary(value,self.trace_expected)
        return value


@contextmanager
def installed_sdk(attention,module,torch,source,source_path,source_sha256,phase):
    require(phase in('prefill','decode'),'Exact SDK phase')
    require(source_path==SOURCE_PATH and source_sha256==SOURCE_SHA and module.__file__==source_path,'Exact pinned SDK module source')
    originals={'sdk_prefill':attention.scaled_qk,'sdk_decode':attention.compute_for_token_gen}
    bindings={}
    for route,original in originals.items():
        require(type(original)is types.MethodType and original.__self__ is attention and
                original.__func__.__globals__ is module.__dict__,'Exact original bound SDK method owner/globals')
        bindings[route]=trace.contract(original,source,source_path,source_sha256,route,attention)
    target='sdk_prefill'if phase=='prefill'else'sdk_decode'
    expected=sdk_descriptor(bindings[target])
    counts=dict(calls=0,other_phase_calls=0,observed_scale_events=0,roles=[],contract=expected,
                mode='full_trace',aten_dispatch_used=False)
    def make(route):
        original=originals[route]
        def factory(fn):
            require(type(fn)is types.MethodType and fn.__self__ is attention and fn.__func__ is original.__func__,
                    'Original SDK callable identity before outer observer')
            @functools.wraps(fn)
            def wrapped(*args,**kwargs):
                capture=base.current()
                if capture is None:return fn(*args,**kwargs)
                if route!=target:counts['other_phase_calls']+=1
                require(route==target,'Other-phase original SDK target invoked')
                require(isinstance(capture,SDKCapture)and capture.route==target and capture.layer_depth==1,
                        'SDK trace inside original outer route and layer')
                require(getattr(capture,'layer0_owner',None)is attention,'Exact driver-bound model.layers[0] attention owner')
                capture.bind_trace(expected)
                with trace.trace_scaled_calls(bindings[route],capture.scaled_local)as state:
                    result=fn(*args,**kwargs)
                roles=[point['role']for point in expected['points']]
                require(state['calls']==1 and state['events']==roles and state['callback_events']==len(roles),
                        'One original SDK method and all ordered scaled locals')
                counts['calls']+=1;counts['observed_scale_events']+=len(roles);counts['roles'].extend(roles)
                return result
            return wrapped
        return factory
    with base.Patches()as patches:
        # Both trace wrappers must precede base.install_sdk: decode's base route
        # wrapper becomes outermost; prefill reaches scaled_qk from that route.
        patches.replace(attention,'scaled_qk',make('sdk_prefill'))
        patches.replace(attention,'compute_for_token_gen',make('sdk_decode'))
        base.install_sdk(patches,attention,module,torch)
        yield counts


def validate_sdk_capture_summary(value,expected):
    route=expected['route'];roles=['qk0']if route=='sdk_prefill'else['qk0','qk1']
    require(route in('sdk_prefill','sdk_decode')and value['complete']is True and value['actual_route']==route and
            value['arm']in('C_Hd_control','C_Hc_history_generation')and value['rank']in(0,1)and
            value['phase']==('prefill'if route=='sdk_prefill'else'decode'),'Exact SDK summary phase/route')
    require(value['original_layer_forward_calls']==1 and value['wrappers_return_original_objects']is True and
            value['scale_observation_schema']==SCHEMA and value['aten_dispatch_used']is False and
            value['trace_contract']==expected and len(value['scale_provenance'])==len(roles),'Complete SDK trace contract')
    require([point['role']for point in expected['points']]==roles,'Ordered SDK descriptor roles')
    for role,operation in zip(roles,value['scale_provenance']):validate_sdk_operation(operation,expected,role)
    wanted={'normalized_input','attention_mask','position_ids','q_projection','k_projection','v_projection',
            'q_rope_input','k_rope_input','v_heads','rope_position_ids','rotary_route','q_post_rope','k_post_rope',
            'rope_cos','rope_sin','context_before_o','attention_output','qk0_unscaled','qk0_scaled','qk0_scale_operation',
            'softmax_input','probabilities_fp32','probabilities_bf16','cache_backing_before_reference','cache_expected_after_reference',
            'returned_current_kv','effective_prior_kv','active_mask','dispatch_flags','route_inputs_'+route}
    if route=='sdk_decode':wanted|={'qk1_unscaled','qk1_scaled','qk1_scale_operation','prior_mask_input','prior_mask_output',
        'softmax_active_input','probabilities_active_fp32','probabilities_active_bf16'}
    else:wanted.add('probabilities_fp32')
    require(wanted<=set(value['events'])and len(value['events'])==len(set(value['events'])),'Every original SDK capture label')
    require([label for label in value['events']if label.endswith('_scale_operation')]==[role+'_scale_operation'for role in roles],
            'Observed ordered SDK operation labels')
    return True
