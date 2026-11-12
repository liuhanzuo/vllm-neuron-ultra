"""Observe actual SDK warmup, flag reset and clear on every DeepSeek buffer."""
import hashlib
import logging
from pathlib import Path

from deepseek_prior_mask_state_v1 import bindings, clear
from deepseek_lifecycle_checks_v2 import STATE_BYTES


def snapshot(model, label, destination, dump=False):
    import torch
    import deepseek_cold_kv_protocol_v2 as p
    initial = bindings(model); rows = []
    folder = destination/'lifecycle-states'/label
    if dump: folder.mkdir(parents=True, exist_ok=False)
    for rank, values in enumerate(model.traced_model.nxd_model.state):
        for name, tensor in sorted(values.items()):
            cpu = tensor.detach().cpu().contiguous(); raw = cpu.view(torch.uint8)
            row = dict(rank=rank, name=name, shape=list(cpu.shape), dtype=str(cpu.dtype),
                       raw_bytes=cpu.numel()*cpu.element_size(), raw_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
                       nonzero_bytes=int(torch.count_nonzero(raw)), nan_elements=int(torch.isnan(cpu).sum()),
                       inf_elements=int(torch.isinf(cpu).sum()))
            if dump:
                path = folder/f'rank-{rank}--{name}.pt'; assert not path.exists()
                torch.save(dict(label=label, rank=rank, name=name, tensor=cpu), path)
                row.update(payload=str(path.relative_to(destination)), payload_sha256=p.sha(path), payload_bytes=path.stat().st_size)
            rows.append(row); del raw, cpu
    assert bindings(model) == initial
    report = dict(bindings=initial, rows=rows, raw_bytes_checked=sum(r['raw_bytes'] for r in rows))
    assert len(rows) == 144 and report['raw_bytes_checked'] == STATE_BYTES
    return report


class Capture(logging.Handler):
    def __init__(self): super().__init__(); self.messages = []
    def emit(self, record): self.messages.append(record.getMessage())


def observe(model, output, execution_guard):
    assert callable(execution_guard)
    execution_guard()
    output=Path(output); destination=output.parent
    assert destination.is_absolute() and destination.resolve()==destination
    assert 'deepseek-cold-kv-diagnostic-v2' not in destination.parts
    import torch
    import deepseek_cold_kv_protocol_v2 as p
    from neuronx_distributed_inference.models import application_base
    assert not output.exists()
    copy = p.read(destination/'state-copy.json'); p.check_copy(copy)
    report = dict(complete=False, passed=False, case=p.expected_case(), snapshots={}, load_skip_warmup=True,
                  copy_controls_completed_before_warmup=True, actual_warmup_calls=0,
                  raw_dump_stages=['before_warmup', 'after_warmup'], historical_raw_byte_origin_proven=False,
                  full_model_qualified=False,
                  scope='Fresh zero states were checked by288fill/288zero controls before this observation. Full before/after-warmup raw dumps and hashes for flag reset/clear preserve all144 bindings; no graph reinitialization. This establishes current lifecycle behavior, not historical raw-byte origin.')
    logger = application_base.logger
    if not isinstance(logger, logging.Logger): logger = logger.logger
    assert isinstance(logger, logging.Logger)
    handler = Capture(); level = logger.level; logger.setLevel(logging.INFO); logger.addHandler(handler)
    try:
        with torch.inference_mode():
            report['snapshots']['before_warmup'] = snapshot(model, 'before_warmup', destination, dump=True)
            assert report['snapshots']['before_warmup']['bindings'] == copy['bindings']
            p.save(output, report)
            assert all(r['nonzero_bytes'] == 0 for r in report['snapshots']['before_warmup']['rows'])
            report['actual_warmup_calls'] = 1; model.warmup()
            report['warmup_messages'] = handler.messages
            report['warmup_completed'] = any('Warmup completed in' in s for s in handler.messages)
            report['warmup_runtime_warnings'] = [s for s in handler.messages if 'Received a ' in s and 'during warmup' in s]
            report['snapshots']['after_warmup'] = snapshot(model, 'after_warmup', destination, dump=True); p.save(output, report)
            report['kv_cache_populated_before_reset'] = bool(model.kv_cache_populated)
            model.reset(); report['kv_cache_populated_after_reset'] = bool(model.kv_cache_populated)
            report['snapshots']['after_reset'] = snapshot(model, 'after_reset', destination); p.save(output, report)
            report['clear_receipt'] = clear(model, copy['bindings'])
            report['snapshots']['after_clear'] = snapshot(model, 'after_clear', destination); p.save(output, report)
        before, after, reset, cleared = [s['rows'] for s in report['snapshots'].values()]
        report.update(complete=True, passed=True,
                      state_tensors_changed_by_warmup=sum(a['raw_sha256'] != b['raw_sha256'] for a, b in zip(before, after)),
                      reset_only_changes_flag=all(a['raw_sha256'] == b['raw_sha256'] for a, b in zip(after, reset)),
                      same144_bindings_preserved=all(s['bindings'] == copy['bindings'] for s in report['snapshots'].values()),
                      raw_dump_bytes=2*STATE_BYTES)
        p.check_lifecycle(report)
    except BaseException as error:
        report.update(passed=False, error=repr(error)); raise
    finally:
        logger.removeHandler(handler); logger.setLevel(level); p.save(output, report)
