"""Single Ray owner with finite EngineCore title dispatch; stdlib source authentication, no import of Ray.

Import is inert. install() is reserved for the independently admitted worker.
The module return event only marks pending; ready() requires natural successful
imports and actual module/function bindings. No environment value is modified.
"""
import hashlib
import json
from pathlib import Path
import sys
import types

import smollm3_observer_invariance_environment_v3 as lineage_policy

require = lineage_policy.require
digest = lineage_policy.digest
SITE = lineage_policy.SITE
KEY = 'RAY_CLIENT_MODE'
WRITER = 'ray._private.client_mode_hook'
WORKER = 'ray._private.worker'
PINS = {
    'ray/_private/client_mode_hook.py': '8837d49d5c22107d20b72807988aafe9865538d9a7213f724f5c472541add007',
    'ray/__init__.py': 'a3845ca44927ca669778ca7489a522f4f5138e8303c7efe6818a193e1b15d376',
    'ray/_private/worker.py': 'ce77b8365900dba15b95c7e2d54146943a7cdff2286ab333871bb85a8442b287',
    'vllm/ray/lazy_utils.py': '5ab09cd6e8cea6235a931eb1a845d38083f1b16caa71702a1d764f6bedd05c42',
    'vllm/engine/arg_utils.py': '2238ade872b5f7478330fa9ca1214389520e08991582090e3e03e2feabcf95ac',
    'vllm/config/parallel.py': '33def3bb46ce10a43df3cd97879a4299b25e48beaceede8203e1941c43c415e1',
    'vllm/executor/ray_utils.py': '8defe57cd0980b9bdf3eac762076a08f117b39903d408d4e8358c2491068072a',
}
MODULES = {
    WRITER: 'ray/_private/client_mode_hook.py', 'ray': 'ray/__init__.py',
    WORKER: 'ray/_private/worker.py', 'vllm.ray.lazy_utils': 'vllm/ray/lazy_utils.py',
    'vllm.engine.arg_utils': 'vllm/engine/arg_utils.py',
    'vllm.config.parallel': 'vllm/config/parallel.py',
    'vllm.executor.ray_utils': 'vllm/executor/ray_utils.py',
}


def observe(actual):
    """Only this nonsecret mode key is summarized, never a whole environment."""
    present = KEY in actual
    value = actual.get(KEY)
    require(not present or type(value) is str, 'Ray mode is an environment string')
    classification = ('absent' if not present else 'zero' if value == '0' else
                      'one' if value == '1' else 'empty' if value == '' else 'other')
    return dict(present=present, classification=classification, exact_zero=present and value == '0',
                length=None if not present else len(value),
                sha256=None if not present else hashlib.sha256(value.encode()).hexdigest())


ABSENT = observe({})
ZERO = observe({KEY: '0'})


def valid_sha(value):
    require(type(value) is str and len(value) == 64 and value != '0'*64 and
            all(c in '0123456789abcdef' for c in value), 'Nonzero SHA256 commitment')


def absent_state():
    return dict(kind='absent', source_receipt_sha256=None, inherited_receipt_sha256=None)


def validate_state(state):
    require(type(state) is dict and set(state) == {'kind', 'source_receipt_sha256', 'inherited_receipt_sha256'},
            'Exact Ray state fields')
    require(state['kind'] in ('absent', 'inherited_zero', 'local_import_zero'), 'Exact Ray state kind')
    if state['kind'] == 'absent':
        require(state == absent_state(), 'Absent has no import or inheritance claim')
    else:
        valid_sha(state['source_receipt_sha256'])
        if state['kind'] == 'inherited_zero':
            require(state['source_receipt_sha256'] == state['inherited_receipt_sha256'], 'Inheritance receipt only')
        elif state['inherited_receipt_sha256'] is not None:
            valid_sha(state['inherited_receipt_sha256'])
    return state


def inherited_state(proof, permit, actual):
    """Root-authenticated dispatch plus earliest child observation, not depth."""
    require(permit['kind'] == 'server' and permit['spawned'] is True, 'Only dispatched server inherits')
    lineage_policy.validate_lineage(permit['environment_lineage'], permit, True)
    edge = permit['environment_lineage']['edges'][-1]
    require(type(proof) is dict and set(proof) == {'schema', 'edge', 'run', 'attempt_id', 'child',
            'parent_state', 'parent_observation', 'child_observation'}, 'Exact Ray inheritance fields')
    require(proof['schema'] == 'smollm3_ray_inheritance_v1' and proof['edge'] == edge and
            proof['run'] == permit['run'] and proof['attempt_id'] == permit['attempt_id'] and
            proof['child'] == lineage_policy.identity(permit), 'Ray inheritance exact lineage/run/nonce')
    validate_state(proof['parent_state'])
    require(proof['parent_state']['kind'] != 'absent' and proof['parent_observation'] == ZERO and
            proof['child_observation'] == ZERO and observe(actual) == ZERO, 'Observed parent and actual child exact zero')
    sha = digest(proof)
    return dict(kind='inherited_zero', source_receipt_sha256=sha, inherited_receipt_sha256=sha)


def code_view(code):
    """Full interpreter-local compiled-module identity; marshal references vary."""
    require(type(code) is types.CodeType, 'Real Python code required')
    fields = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals',
              'co_stacksize', 'co_flags', 'co_code', 'co_names', 'co_varnames', 'co_freevars',
              'co_cellvars', 'co_filename', 'co_name', 'co_firstlineno', 'co_lnotab',
              'co_linetable', 'co_exceptiontable', 'co_qualname')
    out = {name: getattr(code, name) for name in fields if hasattr(code, name)}
    def constant(value):
        if type(value) is types.CodeType: return ('code', code_view(value))
        if type(value) is tuple: return ('tuple', tuple(constant(x) for x in value))
        if type(value) is frozenset: return ('frozenset', tuple(sorted((constant(x) for x in value), key=repr)))
        return (type(value).__name__, repr(value))
    out['co_consts'] = tuple(constant(x) for x in code.co_consts)
    return out


def code_sha(code):
    return hashlib.sha256(repr(code_view(code)).encode()).hexdigest()


def select_code(code, name, line=None):
    found = []
    def visit(item):
        if item.co_name == name and (line is None or item.co_firstlineno == line): found.append(item)
        for child in item.co_consts:
            if type(child) is types.CodeType: visit(child)
    visit(code)
    require(len(found) == 1, 'Unique full-module compiled code selection')
    return found[0]


def module_binding(modules, name, initialized=True):
    module = modules.get(name)
    require(type(module) is types.ModuleType, 'Actual imported module object')
    namespace = module.__dict__; spec = namespace.get('__spec__')
    require(namespace.get('__name__') == name and spec is not None and spec.name == name and
            spec.origin == str(SITE/MODULES[name]) and namespace.get('__file__') == spec.origin,
            'Pinned actual module name/spec/file')
    require(type(getattr(spec, '_initializing', None)) is bool, 'Observed import initialization state')
    if initialized: require(spec._initializing is False, 'Natural import completed')
    return module


def function_binding(module, name, expected, defaults=None, kwdefaults=None):
    function = module.__dict__.get(name)
    require(type(function) is types.FunctionType and code_view(function.__code__) == code_view(expected),
            'Actual function complete source code')
    require(function.__globals__ is module.__dict__ and function.__closure__ is None and
            function.__defaults__ == defaults and function.__kwdefaults__ == kwdefaults,
            'Actual function globals/closure/defaults')
    return function


def validate_receipts(before, after, permit):
    """Independent retained-evidence binding; never substitutes for live trace."""
    require(set(before) == {'schema', 'run', 'attempt_id', 'process', 'permit_sha256', 'stage',
            'lineage_sha256', 'origin', 'observation', 'inherited_receipt_sha256', 'writer_source_sha256',
            'module_code_sha256', 'module_name', 'spec_origin', 'frame_globals_binding'}, 'Exact Ray before fields')
    require(before['schema'] == 'smollm3_ray_import_before_v1' and before['run'] == permit['run'] and
            before['attempt_id'] == permit['attempt_id'] and before['process'] == lineage_policy.identity(permit) and
            before['permit_sha256'] == digest(permit) and before['lineage_sha256'] == digest(permit['environment_lineage']),
            'Ray before exact process/permit/lineage/run')
    require(before['writer_source_sha256'] == PINS[MODULES[WRITER]] and before['module_name'] == WRITER and
            before['spec_origin'] == str(SITE/MODULES[WRITER]) and before['frame_globals_binding'] is True,
            'Ray writer provenance')
    valid_sha(before['module_code_sha256'])
    if permit['spawned']:
        require(before['origin'] == 'inherited_zero' and before['observation'] == ZERO and
                before['stage'] in ('startup', 'import', 'native_ready', 'runner_ready', 'serve') and
                before['inherited_receipt_sha256'] == digest(permit['ray_inheritance']), 'Child separate local import origin')
    else:
        require(before['stage'] == 'api_loading' and before['origin'] == 'root_absent' and
                before['observation'] == ABSENT and before['inherited_receipt_sha256'] is None, 'Root late absent origin')
    require(set(after) == {'schema', 'before_sha256', 'run', 'attempt_id', 'process', 'permit_sha256',
            'normal_import_complete', 'module_binding_verified', 'spec_initializing_false', 'observation',
            'is_client_mode_enabled', 'is_client_mode_enabled_by_default', 'actual_route', 'source_binding_digests'},
            'Exact Ray after fields')
    require(after['schema'] == 'smollm3_ray_import_after_v1' and after['before_sha256'] == digest(before) and
            all(after[key] == before[key] for key in ('run', 'attempt_id', 'process', 'permit_sha256')),
            'Ray complete before/after binding')
    require(all(after[key] is True for key in ('normal_import_complete', 'module_binding_verified', 'spec_initializing_false')) and
            after['observation'] == ZERO and after['is_client_mode_enabled'] is False and
            after['is_client_mode_enabled_by_default'] is False, 'Successful exact Ray import evidence')
    require(after['actual_route'] in ('unresolved', 'lazy_status', 'parallel_config'), 'Actual route enumeration')
    expected = {WRITER: before['module_code_sha256'], WORKER: PINS[MODULES[WORKER]], 'ray': PINS[MODULES['ray']]}
    route_names = {'unresolved': (), 'lazy_status': ('vllm.ray.lazy_utils', 'vllm.engine.arg_utils'),
                   'parallel_config': ('vllm.executor.ray_utils', 'vllm.config.parallel')}
    expected.update({name: PINS[MODULES[name]] for name in route_names[after['actual_route']]})
    require(after['source_binding_digests'] == expected, 'Exact actual route source bindings')
    return dict(kind='local_import_zero', source_receipt_sha256=digest(after),
                inherited_receipt_sha256=before['inherited_receipt_sha256'])


def expected_snapshot_state(permit, stage, before=None, after=None):
    """Snapshots are written on phase entry, before that phase's import body.

    A child local import at stage S therefore cannot appear in S's snapshot;
    every strictly later snapshot must retain the same local receipt. Root's
    ray_ready snapshot is instead written by the dedicated completion method.
    """
    require((before is None)is(after is None),'Local Ray before/after must be paired')
    local=None if before is None else validate_receipts(before,after,permit)
    if not permit['spawned']:
        require(stage in ('clean','startup','import','native_ready','runner_ready','api_loading','ray_ready','serve'),'Known root snapshot stage')
        if stage in ('ray_ready','serve'):
            require(local is not None,'Late root snapshot needs completed import');return local
        return absent_state()
    stages=('clean','startup','import','native_ready','runner_ready','serve')
    require(stage in stages,'Known child snapshot stage')
    inherited=inherited_state(permit['ray_inheritance'],permit,{KEY:'0'})
    if local is None:return inherited
    before_index=stages.index(before['stage'])
    return local if stages.index(stage)>before_index else inherited


class RayImportWitness:
    """One writer execution in this process; environment and receipt owner is context."""
    def __init__(self, context, modules=None):
        self.context = context; self.modules = sys.modules if modules is None else modules
        self.codes = {}; self.before = None; self.after = None; self.returned = False
        self.exception_seen = False; self.writer_globals = None; self.route = 'unresolved'
        self.route_frames = {}; self.busy = False; self.installed_trace = None
        self.module_entries = {}; self.module_returns = set()
        for relative, wanted in PINS.items():
            filename = str(SITE/relative)
            require(context.plan['source_sha256'].get(filename) == wanted, 'Ray chain exact source contract')
            raw = context.source_bytes(Path(filename), wanted)
            require(hashlib.sha256(raw).hexdigest() == wanted, 'Ray source bytes authenticated')
            self.codes[relative] = compile(raw, filename, 'exec', dont_inherit=True, optimize=0)
        require(WRITER not in self.modules and 'ray' not in self.modules and WORKER not in self.modules,
                'No earlier local Ray import or reload')

    def _frame(self, frame, name, function='<module>'):
        relative = MODULES[name]
        if frame.f_code.co_filename != str(SITE/relative) or frame.f_code.co_name != function: return False
        module = module_binding(self.modules, name, initialized=False)
        expected = self.codes[relative] if function == '<module>' else select_code(
            self.codes[relative], function, frame.f_code.co_firstlineno)
        require(code_view(frame.f_code) == code_view(expected) and frame.f_globals is module.__dict__,
                'Actual complete source frame and module globals')
        return True

    def _route(self, frame):
        # Preserve observed order. Importlib frames may separate these entries.
        labels = []; frames = {}; cursor = frame.f_back; depth = 0
        while cursor is not None and depth < 128:
            for name, functions in ((WORKER, ('<module>',)), ('ray', ('<module>',)),
                    ('vllm.ray.lazy_utils', ('is_ray_initialized', 'is_in_ray_actor')),
                    ('vllm.engine.arg_utils', ('create_engine_config',)),
                    ('vllm.executor.ray_utils', ('<module>',)),
                    ('vllm.config.parallel', ('__post_init__',))):
                for function in functions:
                    if self._frame(cursor, name, function):
                        labels.append((name, function)); frames[(name, function)] = cursor
            cursor = cursor.f_back; depth += 1
        prefix = [(WORKER, '<module>'), ('ray', '<module>')]
        for helper in ('is_ray_initialized', 'is_in_ray_actor'):
            route = prefix + [('vllm.ray.lazy_utils', helper), ('vllm.engine.arg_utils', 'create_engine_config')]
            if labels == route: return 'lazy_status', frames
        route = prefix + [('vllm.executor.ray_utils', '<module>'), ('vllm.config.parallel', '__post_init__')]
        if labels == route: return 'parallel_config', frames
        return 'unresolved', {}

    def begin(self, frame):
        require(self.before is None and self.after is None, 'Single writer execution; reload forbidden')
        require(self._frame(frame, WRITER), 'Pinned writer module entry')
        state = validate_state(self.context.ray_state)
        require(self.context.kind == 'server', 'Client never imports Ray')
        if not self.context.spawned:
            require(self.context.environment_stage == 'api_loading' and state == absent_state() and
                    observe(self.context.actual_environment()) == ABSENT, 'Root only absent late API import')
        else:
            require(state['kind'] == 'inherited_zero' and observe(self.context.actual_environment()) == ZERO,
                    'Child local import starts from proven inherited zero')
        self.context.poll()
        self.route, self.route_frames = self._route(frame)
        permit = self.context.permit
        row = dict(schema='smollm3_ray_import_before_v1', run=permit['run'], attempt_id=permit['attempt_id'],
                   process=lineage_policy.identity(permit), permit_sha256=digest(permit), stage=self.context.environment_stage,
                   lineage_sha256=digest(self.context.lineage), origin='inherited_zero' if self.context.spawned else 'root_absent',
                   observation=observe(self.context.actual_environment()), inherited_receipt_sha256=state['inherited_receipt_sha256'],
                   writer_source_sha256=PINS[MODULES[WRITER]], module_code_sha256=code_sha(frame.f_code),
                   module_name=WRITER, spec_origin=str(SITE/MODULES[WRITER]), frame_globals_binding=True)
        self.context.record('ray-before-'+str(permit['pid'])+'.json', row)
        self.writer_globals = frame.f_globals; self.before = row

    def event(self, frame, event, argument):
        if event not in ('call', 'return', 'exception'): return
        for name in ('ray', WORKER):
            if frame.f_code.co_filename == str(SITE/MODULES[name]) and frame.f_code.co_name == '<module>':
                if event == 'call':
                    require(self._frame(frame, name), 'Actual pinned enclosing Ray module')
                    require(name not in self.module_entries, 'Enclosing Ray module reload forbidden')
                    self.module_entries[name] = frame.f_globals
                elif event == 'return':
                    require(self.module_entries.get(name) is frame.f_globals, 'Same enclosing module return')
                    self.module_returns.add(name)
        if frame.f_code.co_filename != str(SITE/MODULES[WRITER]) or frame.f_code.co_name != '<module>': return
        if event == 'call': self.begin(frame)
        elif self.before is not None:
            require(frame.f_globals is self.writer_globals and self._frame(frame, WRITER), 'Same writer execution')
            if event == 'exception': self.exception_seen = True
            elif event == 'return':
                require(not self.exception_seen, 'Exceptional writer return cannot certify import')
                self.returned = True

    def ready(self, committed=False):
        require(self.before is not None and self.returned and not self.exception_seen and
                ((self.after is not None) if committed else self.after is None),
                'Normal pending writer execution required')
        writer = module_binding(self.modules, WRITER)
        require(writer.__dict__ is self.writer_globals, 'Successful import same executed namespace')
        worker = module_binding(self.modules, WORKER); ray = module_binding(self.modules, 'ray')
        require(self.module_returns == {'ray', WORKER} and self.module_entries.get('ray') is ray.__dict__ and
                self.module_entries.get(WORKER) is worker.__dict__, 'Complete actual enclosing module executions')
        require(observe(self.context.actual_environment()) == ZERO, 'Actual Ray setter exact zero')
        require(writer.__dict__.get('is_client_mode_enabled') is False and
                writer.__dict__.get('is_client_mode_enabled_by_default') is False, 'Both Ray client flags disabled')
        code = self.codes[MODULES[WRITER]]
        hook = function_binding(writer, 'client_mode_hook', select_code(code, 'client_mode_hook', 95),
                                defaults=(None,), kwdefaults={'local_only_kwargs': ()})
        function_binding(writer, '_explicitly_enable_client_mode', select_code(code, '_explicitly_enable_client_mode'))
        function_binding(writer, '_explicitly_disable_client_mode', select_code(code, '_explicitly_disable_client_mode'))
        require(worker.__dict__.get('client_mode_hook') is hook, 'Actual worker binding to executed writer')
        bindings = {WRITER: code_sha(code), WORKER: PINS[MODULES[WORKER]], 'ray': PINS[MODULES['ray']]}
        if self.route == 'lazy_status':
            lazy = module_binding(self.modules, 'vllm.ray.lazy_utils')
            args = module_binding(self.modules, 'vllm.engine.arg_utils')
            for name in ('is_ray_initialized', 'is_in_ray_actor'):
                function = function_binding(lazy, name, select_code(self.codes[MODULES['vllm.ray.lazy_utils']], name))
                require(args.__dict__.get(name) is function, 'Actual arg_utils bound lazy helper')
            bindings.update({name: PINS[MODULES[name]] for name in ('vllm.ray.lazy_utils', 'vllm.engine.arg_utils')})
        elif self.route == 'parallel_config':
            utils = module_binding(self.modules, 'vllm.executor.ray_utils')
            function_binding(utils, 'ray_is_available', select_code(self.codes[MODULES['vllm.executor.ray_utils']], 'ray_is_available'))
            require(utils.__dict__.get('ray') is ray, 'Actual ray_utils Ray module')
            caller = self.route_frames[('vllm.config.parallel', '__post_init__')]
            require(caller.f_locals.get('ray_utils') is utils, 'Actual ParallelConfig imported ray_utils binding')
            bindings.update({name: PINS[MODULES[name]] for name in ('vllm.executor.ray_utils', 'vllm.config.parallel')})
        return dict(schema='smollm3_ray_import_after_v1', before_sha256=digest(self.before),
                    run=self.before['run'], attempt_id=self.before['attempt_id'], process=self.before['process'],
                    permit_sha256=self.before['permit_sha256'], normal_import_complete=True, module_binding_verified=True,
                    spec_initializing_false=True, observation=ZERO, is_client_mode_enabled=False,
                    is_client_mode_enabled_by_default=False, actual_route=self.route, source_binding_digests=bindings)

    def commit(self, after):
        require(self.after is None and after == self.ready(), 'Single authenticated completion')
        self.after = after
        self.route_frames = {}

    def check_committed(self):
        require(self.after is not None, 'Committed Ray import required')
        # Actual route frame objects are dropped after commit. Their prior
        # complete binding remains in the receipt; all core bindings stay live.
        writer = module_binding(self.modules, WRITER)
        worker = module_binding(self.modules, WORKER)
        module_binding(self.modules, 'ray')
        require(writer.__dict__ is self.writer_globals and observe(self.context.actual_environment()) == ZERO and
                writer.__dict__.get('is_client_mode_enabled') is False and
                writer.__dict__.get('is_client_mode_enabled_by_default') is False, 'Committed Ray state unchanged')
        code = self.codes[MODULES[WRITER]]
        hook = function_binding(writer, 'client_mode_hook', select_code(code, 'client_mode_hook', 95),
                                defaults=(None,), kwdefaults={'local_only_kwargs': ()})
        function_binding(writer, '_explicitly_enable_client_mode', select_code(code, '_explicitly_enable_client_mode'))
        function_binding(writer, '_explicitly_disable_client_mode', select_code(code, '_explicitly_disable_client_mode'))
        require(worker.__dict__.get('client_mode_hook') is hook, 'Committed worker writer binding unchanged')
        require(self.installed_trace is None or sys.gettrace() is self.installed_trace, 'Ray source witness remains installed')

    def install(self):
        require(sys.gettrace() is None, 'No previous Python trace to replace')
        # Construct once after source-authenticated witness initialization.
        # Core module events remain first, including after commit: reload checks
        # must never be skipped by the committed-state fast return.
        core_filenames = frozenset(str(SITE/MODULES[name]) for name in (WRITER, WORKER, 'ray'))
        chain_filenames = frozenset(str(SITE/value) for value in MODULES.values())
        # A single owner dispatches finite EngineCore title events as well as
        # the original Ray events. No second sys.settrace owner is installed.
        title_witness = getattr(self.context, 'title_witness', None)
        if title_witness is not None:
            from smollm3_enginecore_title_hook_v3 import EngineCoreTitleHook
            require(type(title_witness) is EngineCoreTitleHook, 'Exact finite title witness')
        def trace(frame, event, argument):
            if self.busy: return None
            if title_witness is not None and title_witness.needs_event(frame):
                self.busy = True
                try: title_witness.event(frame, event, argument)
                finally: self.busy = False
                return trace
            is_module = frame.f_code.co_filename in core_filenames and frame.f_code.co_name == '<module>'
            if is_module:
                self.busy = True
                try: self.event(frame, event, argument)
                finally: self.busy = False
                return trace
            if self.after is not None: return None
            # Only trace frames in the authenticated import chains. A line after
            # the natural import may commit pending before later context.poll.
            if self.before is not None and self.after is None and frame.f_code.co_filename in chain_filenames:
                route_resumed = self.route == 'lazy_status' and frame in self.route_frames.values() and frame.f_code.co_name in ('is_ray_initialized', 'is_in_ray_actor')
                route_resumed = route_resumed or self.route == 'parallel_config' and frame is self.route_frames.get(('vllm.config.parallel', '__post_init__')) and 'ray_utils' in frame.f_locals
                route_resumed = route_resumed or self.route == 'unresolved'
                if event == 'line' and route_resumed and self.returned and all(type(self.modules.get(name)) is types.ModuleType and
                        getattr(self.modules[name].__dict__.get('__spec__'), '_initializing', True) is False
                        for name in (WRITER, WORKER, 'ray')):
                    self.busy = True
                    try: self.context.complete_ray_import()
                    finally: self.busy = False
                return trace
            if event == 'call' and frame.f_code.co_filename in chain_filenames:
                return trace
            return None
        self.installed_trace = trace
        if title_witness is not None: title_witness.set_trace_owner(trace)
        sys.settrace(trace)
        return self
