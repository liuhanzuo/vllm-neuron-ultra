"""Pure conservative module-body import graph; never imports inspected code.

Scope is one explicit package plus its entrypoint. External imports are report
boundaries, not a claimed complete Python/native dependency closure. Function
bodies are reported separately; decorators/defaults and dynamic imports remain
limitations. Both branches of module/class control flow are conservatively
visited without attempting to execute conditions.
"""
import ast


def require(value, message):
    if not value:
        raise ValueError(message)


def inspect_imports(source):
    tree = ast.parse(source, feature_version=(3, 10))
    eager, deferred, dynamic = [], [], []

    class Scan(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0

        def function(self, node):
            for value in (*node.decorator_list, *node.args.defaults,
                          *(x for x in node.args.kw_defaults if x is not None)):
                self.visit(value)
            self.depth += 1
            for value in node.body:
                self.visit(value)
            self.depth -= 1

        visit_FunctionDef = function
        visit_AsyncFunctionDef = function

        def visit_Lambda(self, node):
            for value in (*node.args.defaults, *(x for x in node.args.kw_defaults if x is not None)):
                self.visit(value)
            self.depth += 1
            self.visit(node.body)
            self.depth -= 1

        def visit_Import(self, node):
            (deferred if self.depth else eager).append(dict(kind='import', level=0,
                module=None, names=[x.name for x in node.names], line=node.lineno))

        def visit_ImportFrom(self, node):
            (deferred if self.depth else eager).append(dict(kind='from', level=node.level,
                module=node.module, names=[x.name for x in node.names], line=node.lineno))

        def visit_Call(self, node):
            f = node.func
            if isinstance(f, ast.Name) and f.id in ('__import__', 'exec', 'eval') or (
                    isinstance(f, ast.Attribute) and f.attr in ('import_module', 'load_module', 'exec_module')):
                dynamic.append(dict(line=node.lineno, expression=ast.unparse(f),
                                    in_function_body=self.depth > 0))
            self.generic_visit(node)

    Scan().visit(tree)
    return dict(module_body_imports=eager, function_body_imports=deferred,
                dynamic_import_or_execution_calls=dynamic)


def graph(entry_source, modules, namespaces, package='libneuronxla'):
    require(package in modules and modules[package]['is_package'] is True,
            'Explicit root package source')
    require(type(namespaces) is set and not (set(modules) & namespaces), 'Disjoint namespace directories')
    for name, row in modules.items():
        require(name == package or name.startswith(package+'.'), 'Only selected package modules')
        require(set(row) == {'source', 'is_package'} and type(row['source']) is str
                and type(row['is_package']) is bool, 'Explicit source-only module descriptor')
    known = set(modules) | namespaces
    pending, visited, edges, external, scans = ['<entry>'], set(), [], [], {}

    def local(name):
        return name == package or name.startswith(package+'.')

    def add(importer, target, line, reason):
        if not local(target):
            external.append(dict(importer=importer, target=target, line=line, reason=reason))
            return
        require(target in known, 'Unresolved selected-package import: '+target)
        parts = target.split('.')
        for n in range(1, len(parts)+1):
            name = '.'.join(parts[:n])
            require(name in known, 'Missing implicit parent package: '+name)
            edge = dict(importer=importer, target=name, line=line,
                        reason=reason if name == target else 'implicit_parent_package')
            if edge not in edges:
                edges.append(edge)
            if name not in visited and name not in pending:
                pending.append(name)

    while pending:
        name = pending.pop(0)
        if name in visited:
            continue
        visited.add(name)
        if name in namespaces:
            scans[name] = dict(namespace_package=True, module_body_imports=[], function_body_imports=[],
                               dynamic_import_or_execution_calls=[])
            continue
        source = entry_source if name == '<entry>' else modules[name]['source']
        current = '' if name == '<entry>' else name if modules[name]['is_package'] else name.rpartition('.')[0]
        scanned = inspect_imports(source)
        scans[name] = scanned
        for item in scanned['module_body_imports']:
            if item['kind'] == 'import':
                for target in item['names']:
                    add(name, target, item['line'], 'explicit_import')
                continue
            if item['level']:
                parts = current.split('.') if current else []
                require(len(parts) >= item['level'], 'Relative import beyond package')
                stem = '.'.join(parts[:len(parts)-item['level']+1])
                target = stem+('.'+item['module'] if item['module'] else '')
            else:
                target = item['module'] or ''
            require(target, 'Explicit absolute import target')
            add(name, target, item['line'], 'from_module')
            for member in item['names']:
                if member == '*':
                    external.append(dict(importer=name, target=target+'.*', line=item['line'],
                                         reason='star_export_resolution_unproven'))
                    continue
                candidate = target+'.'+member
                if local(target) and candidate in known:
                    add(name, candidate, item['line'], 'from_submodule_candidate')
                elif not local(target):
                    external.append(dict(importer=name, target=candidate, line=item['line'],
                                         reason='external_member_or_submodule_unresolved'))
    return dict(reachable_python_modules=sorted(visited & set(modules)),
        reachable_namespace_packages=sorted(visited & namespaces),
        local_import_edges=edges, external_import_boundaries=external,
        module_scans=scans, native_or_package_code_executed=False,
        complete_runtime_dependency_closure_proven=False,
        limitations=[
            'Conservative syntax graph: both branches and class bodies visited, not runtime observations.',
            'Function bodies are listed separately and not followed; calls during import can execute them.',
            'Aliases, import hooks, star exports, annotations and arbitrary dynamic execution are not fully resolved.',
            'External member names may be attributes or submodules; external packages and native extensions remain separate review boundaries.'
        ])
