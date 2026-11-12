"""Unbound runtime interface for unchanged device scientific bodies.

A future root-owned runtime must implement every method, bind actual full CPU
success/idle/history/resources and source hashes, then call a leaf. This module
creates no admission, paths, lock, service or current runnable binding.
"""
from abc import ABC,abstractmethod
from pathlib import Path
from lfm3_typed_slot_writer_contract_v1 import required_gates
from slot_cache_nki05_typed_jit_v1 import DEVICE_SOFTWARE,make_device_kernel
from slot_cache_nki05_typed_torch_v1 import make_typed_writer

ARTIFACT_SCHEMA='lfm3-nki05-typed-device-artifact-v1'
VARIANT='typed_selector_nki05_v1'
CONTEXT4096_REQUIRED=True
FULL_MODEL_GATES=required_gates()
RUNTIME_ADMITTED=False
COMPILE_PEAK_BYTES=None
COMPILE_ARTIFACT_CAP_BYTES=None
REQUIRED_GATES=dict(cpu_regression=1648,cpu_extended=2068,raw_device_graphs=4,raw_device_checks=36,
    regression_device_graphs=9,regression_device_checks=43,regression_alias_graphs=12,regression_alias_checks=60,
    extended_device_graphs=42,extended_device_checks=2520,extended_alias_graphs=24,extended_alias_checks=192,
    fp32_alias_required=True,int64_oob_before_narrowing_required=True,one_row_original_dispatch_required=True,
    cold4096_required=True,long4096_required=True,slot_reuse4096_required=True,serving_graph4096_required=True)
PHASES=('raw_device','regression_device','regression_alias','extended_device','extended_alias',
        'fp32_alias','wrapper_controls','context4096_device','context4096_alias')


class DeviceContext(ABC):
    """Actual owning runtime implements authority and bounded IO, not this source-only leaf."""
    @abstractmethod
    def authorize(self,phase):
        """Authenticate source/software, CPU actual exit+full audit, current idle/history,
        prerequisites and owned process before imports; bound compile/artifact budgets.
        Must raise on failure; return exactly True while the actual phase remains owned.
        """
        raise NotImplementedError
    @abstractmethod
    def prerequisite(self,name):
        """Return independently authenticated full CPU or preceding device gate metadata."""
        raise NotImplementedError
    @abstractmethod
    def paths(self,phase):
        """Return (original source root, exclusive fresh phase output root)."""
        raise NotImplementedError
    @abstractmethod
    def sources(self):
        """Full actual leaf/kernel/SDK scientific source digest map, never old artifact pins."""
        raise NotImplementedError
    @abstractmethod
    def save(self,path,value,phase):
        """Owned atomic bounded JSON save, always below this fresh output root."""
        raise NotImplementedError
    @abstractmethod
    def hash_artifact(self,path,phase):
        """Bounded streaming digest of this admitted artifact; never read old artifacts."""
        raise NotImplementedError
    @abstractmethod
    def record_artifact(self,path,phase,case):
        """Save and return new receipt with schema, variant, software, source_sha256,
        phase, path, case and sha256. Runtime must authenticate and bound all IO.
        Reference/candidate label and complete actual shapes/dtypes are in case.
        """
        raise NotImplementedError

    def protocol(self,phase):
        # Immutable matrix functions are source-pinned, never caller-supplied.
        from lfm3_nki05_device_matrix_v1 import BoundMatrix
        self.require_phase(phase)
        return BoundMatrix(self,phase)

    def artifact(self,path,phase,case):
        self.require_phase(phase)
        root=Path(self.paths(phase)[1]).resolve()
        actual=Path(path).resolve()
        if not actual.is_relative_to(root):
            raise RuntimeError('Artifact must be under the admitted fresh output root')
        receipt=self.record_artifact(actual,phase,case)
        expected=dict(schema=ARTIFACT_SCHEMA,variant=VARIANT,software=DEVICE_SOFTWARE,
            source_sha256=self.sources(),phase=phase,path=str(actual),case=case,
            sha256=self.hash_artifact(actual,phase))
        if not isinstance(receipt,dict) or any(receipt.get(k)!=v for k,v in expected.items()):
            raise RuntimeError('Distinct exact NKI0.5 artifact identity required')
        self.require_phase(phase)
        return receipt

    def writer(self,phase):
        self.require_phase(phase)
        admit=lambda:self.authorize(phase)
        return make_typed_writer(make_device_kernel(admit),admit)

    def require_phase(self,phase):
        if not __debug__ or phase not in PHASES or self.authorize(phase) is not True:
            raise RuntimeError('Explicit owned device phase required; no default admission')


def require_context(context,phase):
    if not isinstance(context,DeviceContext):raise RuntimeError('Future DeviceContext implementation required')
    context.require_phase(phase)
    require_cpu_summary(context.prerequisite('full_cpu'))
    return tuple(Path(value) for value in context.paths(phase))


def require_cpu_summary(proof):
    """Necessary summary checks; context must independently authenticate every batch,
    actual outer exit, warning receipts, drained owners/lock and fresh handoff.
    This check alone does not provide runtime admission.
    """
    expected={stage:dict(cases=3716,batches=291)
        for stage in ('original_fixture','new_fixture','candidate')}
    expected['controls']=dict(cases=1,batches=1)
    if not isinstance(proof,dict) or proof.get('schema')!='lfm3_nki05_full_cpu_result_v1':
        raise RuntimeError('Actual full CPU successor proof required')
    if proof.get('stage_counts')!=expected:
        raise RuntimeError('Retain all three full 3716-case matrices and actual control')
    if any(proof.get(k) is not True for k in ('complete','passed','cpu_full_gate_passed',
            'simulator_writeback_workaround_required','original_simulator_failure_preserved')):
        raise RuntimeError('Original and modified simulator evidence required')
    if any(proof.get(k) is not False for k in ('device_behavior_proven','alias_gate_passed',
            'full_model_qualified','new_speedup_claim','automatic_successor_launch')):
        raise RuntimeError('CPU result cannot promote device/model/performance status')

