"""Native160 trace integration draft; no IO/native imports or admission.

Original full observer is reused unchanged. Only the scaled-local observation
is supplied by the separately accepted Python trace helper. SDK88 is future.
"""
from contextlib import contextmanager
import functools
import copy
import minicpm5_layer0_capture_observer_v2 as base
from research import minicpm5_python_scale_trace_candidate_20261107_v2 as trace

require=base.require
SCHEMA='minicpm5_python_ast_scale_observation_v1'
ARMS=('unobserved','previous213_hooks','noop_wrappers','noop_trace','full_trace')
BASE_KEYS={'observation','scaled_value_observed','scaled_value_recomputed','source','function',
           'assignment_line','event_line','local_name','operator','operator_identity_basis',
           'scalar','scalar_origin','aten_operator_observed','route','line_metadata'}
OPERAND_BASIS='same_pinned_assignment_after_original_matmul_return'


def descriptor(binding):
    require(binding['route']=='hf_eager' and len(binding['events'])==1,'Native160 exact route')
    point=next(iter(binding['events'].values()))
    require(point['role']=='qk0','Original QK0 role')
    return dict(source=copy.deepcopy(binding['source']),function=binding['function'],
                line_metadata=copy.deepcopy(binding['line_metadata']),point=copy.deepcopy(point))


def validate_operation(value,expected,bridged=True):
    """New explicit schema; old aten metadata is never translated or accepted."""
    require(isinstance(value,dict),'Scale provenance mapping')
    keys=BASE_KEYS|({'operand_binding_basis','schema'} if bridged else set())
    require(set(value)==keys,'Exact Python AST provenance keys')
    require(value['observation']=='actual_python_local_after_assignment_before_next_statement' and
            value['scaled_value_observed'] is True and value['scaled_value_recomputed'] is False and
            value['aten_operator_observed'] is False,'Actual local, no fabricated aten evidence')
    require(value['operator']=='python.multiply' and value['operator_identity_basis']=='pinned_source_AST' and
            value['scalar_origin']=='actual_function_argument' and type(value['scalar']) in (int,float) and
            value['scalar']==128**-.5,'Exact native Python scaling provenance')
    point=expected['point']
    require(value['source']==expected['source'] and value['function']==expected['function'] and
            value['line_metadata']==expected['line_metadata'],'Authenticated original code/line provenance')
    require(value['route']=='hf_eager' and value['assignment_line']==point['assignment_line'] and
            value['event_line']==point['event_line'] and value['local_name']==point['variable'],
            'Actual assigned local and exact observation boundary')
    if bridged:
        require(value['schema']==SCHEMA and value['operand_binding_basis']==OPERAND_BASIS,
                'Explicit successor operand provenance')
    return True


class Capture(base.Capture):
    def bind_trace(self,expected):
        require(self.arm=='HF_native' and not hasattr(self,'trace_expected'),'Native trace binding once')
        self.trace_expected=copy.deepcopy(expected);self.scale_provenance=[]

    def observe_scale(self,*args,**kwargs):
        raise RuntimeError('Aten dispatch evidence is forbidden in Python trace successor')

    def scaled_local(self,role,actual,provenance):
        require(hasattr(self,'trace_expected') and role=='qk0','Authenticated native trace active')
        validate_operation(provenance,self.trace_expected,bridged=False)
        # Accepted bridge observes the same actual object; no recomputation.
        trace.capture_bridge(self,role,actual,provenance)

    def emit(self,label,value,selection=None):
        if label.endswith('_scale_operation'):
            require(label=='qk0_scale_operation' and hasattr(self,'trace_expected'),'Native scale scope')
            value=dict(value,schema=SCHEMA)
            validate_operation(value,self.trace_expected)
            require(not self.scale_provenance,'Exactly one observed scaled local')
            self.scale_provenance.append(copy.deepcopy(value))
        return super().emit(label,value,selection)

    def complete(self):
        result=super().complete()
        require(hasattr(self,'trace_expected') and len(self.scale_provenance)==1,'Complete actual Python scale evidence')
        validate_operation(self.scale_provenance[0],self.trace_expected)
        result.update(scale_observation_schema=SCHEMA,scale_provenance=copy.deepcopy(self.scale_provenance),
                      trace_contract=copy.deepcopy(self.trace_expected),aten_dispatch_used=False)
        return result


@contextmanager
def installed_native(attention,module,torch,source,source_path,source_sha256,full):
    """Fresh outer patch scope; trace is inside the exact original eager call.

    Caller authenticates all original module functions and source pins before
    installation. The helper additionally checks original eager code/line bytes.
    No trace spans a whole model or another layer. full=False is noop_trace.
    """
    require(type(full)is bool and attention.layer_idx==0,'Exact native layer0 trace mode')
    original=module.eager_attention_forward
    binding=trace.contract(original,source,source_path,source_sha256,'hf_eager',attention)
    expected=descriptor(binding)
    counts=dict(calls=0,observed_scale_events=0,noop_scale_events=0,contract=expected,
                mode='full_trace' if full else 'noop_trace',aten_dispatch_used=False)
    def make(fn):
        require(fn is original,'Original eager callable identity')
        @functools.wraps(fn)
        def wrapped(*args,**kwargs):
            bound=base.bind(fn,args,kwargs)
            if bound['module']is not attention:return fn(*args,**kwargs)
            capture=base.current()
            if full:
                require(isinstance(capture,Capture) and capture.route=='hf_eager',
                        'Trace must run inside original full observer route')
                capture.bind_trace(expected);callback=capture.scaled_local
            else:
                require(capture is None,'Noop trace must not capture tensors');callback=None
            with trace.trace_scaled_calls(binding,callback)as state:
                out=fn(*args,**kwargs)
            require(state['calls']==1 and state['events']==['qk0'] and
                    state['callback_events']==(1 if full else 0),'Exact one original eager scale boundary')
            counts['calls']+=1
            counts['observed_scale_events' if full else 'noop_scale_events']+=1
            return out
        return wrapped
    with base.Patches()as patches:
        # Install this first: original observer's route wrapper becomes outer,
        # so its unscaled matmul snapshot exists before the local trace event.
        patches.replace(module,'eager_attention_forward',make)
        if full:base.install_native(patches,attention,module,torch)
        yield counts


def validate_capture_summary(value,expected):
    require(value['complete'] is True and value['actual_route']=='hf_eager' and
            value['original_layer_forward_calls']==1 and value['wrappers_return_original_objects'] is True,
            'Original complete capture boundary')
    require(value['scale_observation_schema']==SCHEMA and value['aten_dispatch_used'] is False and
            value['trace_contract']==expected and len(value['scale_provenance'])==1,'Successor explicit schema')
    validate_operation(value['scale_provenance'][0],expected)
    # base.Capture.complete enforces the full label set; independent metadata
    # validation also requires every native label, never just scaled QK.
    wanted={'normalized_input','attention_mask','position_ids','q_projection','k_projection','v_projection',
            'q_rope_input','k_rope_input','v_heads','rope_position_ids','rotary_route','q_post_rope','k_post_rope',
            'rope_cos','rope_sin','context_before_o','attention_output','qk0_unscaled','qk0_scaled',
            'qk0_scale_operation','softmax_input','probabilities_bf16','cache_before','cache_after',
            'native_effective_key','native_effective_value','probabilities_fp32'}
    require(wanted<=set(value['events']) and len(value['events'])==len(set(value['events'])),
            'Full original native capture labels')
    return True
