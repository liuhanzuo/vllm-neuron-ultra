"""One outer312 source pin and honest pre-JSON execution observation.

Inert, import-free source for a future exact frozen literal. No filesystem or
process operations of its own. Source reads use the existing bounded Entry.
Do not import this local module before Entry.plan is available; embed its exact
reviewed source alongside the existing helper, before child.main is called.
"""

PIN = {
    'path': '/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/types.py',
    'sha256': '345474ef027a1273f353da9bdc1f7c18f65335e72e681bcc0376774cc51f2405',
    'bytes': 10993,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def provider_snapshot(runtime):
    module = runtime.modules.get('types')
    spec = getattr(module, '__spec__', None)
    return dict(present=module is not None,
                module_file=getattr(module, '__file__', None),
                spec_name=getattr(spec, 'name', None),
                spec_origin=getattr(spec, 'origin', None),
                loader_type=type(getattr(module, '__loader__', None)).__name__,
                loader_is_spec_loader=getattr(module, '__loader__', None) is getattr(spec, 'loader', None))


class TypesSourceObservation:
    """Keep loaded-at-entry evidence separate from execution through our loader."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.initial_module = runtime.modules.get('types')
        self.initial_snapshot = provider_snapshot(runtime)
        self.entry = self.external = self.finder = self.loader_type = None
        self.executed_module = self.execution_receipt = None
        self.after_json_snapshot = None
        self.finished = False

    def _source(self):
        raw, identity = self.entry.read_source(PIN['path'], PIN['sha256'], PIN['bytes'])
        require(len(raw) == PIN['bytes'], 'Exact reviewed outer types source size')
        self.entry.source_bytes += len(raw)
        require(self.entry.source_bytes <= self.source_total_cap, 'Original whole-source byte cap')
        return raw, dict(sha256=PIN['sha256'], bytes=len(raw), identity=identity)

    def _provider(self, module, loader):
        spec = getattr(module, '__spec__', None)
        require(module is not None and getattr(module, '__name__', None) == 'types'
                and getattr(module, '__file__', None) == PIN['path']
                and getattr(spec, 'name', None) == 'types'
                and getattr(spec, 'origin', None) == PIN['path']
                and getattr(module, '__loader__', None) is loader
                and getattr(spec, 'loader', None) is loader,
                'Exact loaded types module/source/spec/loader correspondence')

    def install_before_json(self, entry, external, source_total_cap):
        require(self.entry is None and self.runtime.modules.get('types') is self.initial_module,
                'One installation; preserve earliest types module identity')
        require(source_total_cap == 16 * 2**20, 'Unchanged original whole-source byte cap')
        require(self.runtime.meta_path and self.runtime.meta_path[0] is entry.finder,
                'Wrap the original first source finder')
        self.entry, self.external = entry, external
        self.source_total_cap = source_total_cap
        tracker, previous = self, entry.finder

        class TypesLoader(external.SourceFileLoader):
            def get_code(self, name):
                require(name == 'types' and self.path == PIN['path'], 'Only the exact types source')
                raw, row = tracker._source()
                code = compile(raw, PIN['path'], 'exec', dont_inherit=True, optimize=0)
                self.pending_receipt = row
                return code

            def exec_module(self, module):
                require(tracker.initial_module is None and tracker.executed_module is None,
                        'Never reload or replace an already loaded types module')
                tracker._provider(module, self)
                entry.check()
                super().exec_module(module)
                require(tracker.runtime.modules.get('types') is module,
                        'Same module remains registered after complete execution')
                tracker._provider(module, self)
                entry.check()
                # get_code alone is not an execution receipt. Record only after
                # SourceFileLoader.exec_module returns successfully.
                entry.sources[PIN['path']] = dict(self.pending_receipt)
                tracker.execution_receipt = dict(self.pending_receipt)
                tracker.executed_module = module

        class TypesFinder:
            def find_spec(self, name, path=None, target=None):
                found = previous.find_spec(name, path, target)
                if name != 'types':
                    return found
                require(found is None, 'No different earlier types provider')
                spec = external.PathFinder.find_spec(name, path, target)
                require(spec is not None and spec.origin == PIN['path']
                        and type(spec.loader) is external.SourceFileLoader,
                        'Only the reviewed ordinary types source provider')
                spec.loader = TypesLoader(name, PIN['path'])
                return spec

        self.loader_type = TypesLoader
        self.finder = TypesFinder()
        self.runtime.meta_path[0] = self.finder
        entry.finder = self.finder
        entry.check()

    def after_json(self):
        require(self.entry is not None and self.after_json_snapshot is None,
                'One explicit checkpoint after successful JSON import')
        self.after_json_snapshot = provider_snapshot(self.runtime)

    def finish(self, candidate_row):
        require(not self.finished and self.after_json_snapshot is not None,
                'Complete observation only after JSON checkpoint')
        require({key: candidate_row[key] for key in PIN} == PIN,
                'Early literal pin equals the later authenticated candidate table')
        module = self.runtime.modules.get('types')
        if self.executed_module is not None:
            require(module is self.executed_module
                    and type(getattr(module, '__loader__', None)) is self.loader_type
                    and self.entry.sources.get(PIN['path']) == self.execution_receipt,
                    'Same completed pinned types execution and retained receipt')
        else:
            require(self.initial_module is not None and module is self.initial_module
                    and type(getattr(module, '__loader__', None)) is self.external.SourceFileLoader
                    and PIN['path'] not in self.entry.sources,
                    'Preexisting module stays preexisting; never fabricate its execution receipt')
        self._provider(module, module.__loader__)
        _, current_source = self._source()
        self.entry.check()
        self.finished = True
        executed = self.executed_module is not None
        return dict(schema='minicpm5_outer_types_source_observation_v1',
                    first_checkpoint=self.initial_snapshot,
                    after_json=self.after_json_snapshot,
                    final_provider=provider_snapshot(self.runtime),
                    initial_load_timing='before_first_checkpoint' if self.initial_module is not None else 'after_first_checkpoint',
                    current_source_reference=dict(PIN), current_source_identity=current_source['identity'],
                    current_source_bytes_verified=True, loaded_provider_metadata_matches=True,
                    executed_through_pinned_loader=executed,
                    execution_receipt=dict(self.execution_receipt) if executed else None,
                    original_execution_provenance_gap=not executed,
                    interpreter_startup_cause_established=False,
                    preexisting_module_reloaded=False,
                    accepted_native_authority_created=False,
                    full_model_qualified=False, new_speedup_claim=False)
