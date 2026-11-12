# MiniCPM5 formal r4: failure before SDK start

This is a saved-output/source review, not the root's physical drain audit.
Root reported actual session 95781, terminal 270de1, exit 1. The independently
read launcher receipt records PID 855446 and actual `Popen.wait()` return 1.
The final 2,293-byte log ends at outer v7 line 163, observed adapter v4 line 79,
original diagnostic line 275, with `ValueError: Actual original authenticated
outer and exclusive admission`. Output contains only 43,273-byte admission.json.

The first conjunct fails deterministically on Python tuple/list equality:

1. Frozen literal v6 `Entry.read_source`, line 280, creates a tuple of seven
   file identity integers and returns it at line 305. `TypesSourceObservation`
   keeps that tuple in `_source` (lines 517–521) and `finish` (lines 612–623).
2. Outer v7 line 150 deep-copies the types observation into the in-memory
   admission. The tuple survives this copy in both
   `source_entry_binding.outer_types_source_observation.current_source_identity`
   and `.execution_receipt.identity`.
3. Saving and rereading through the original literal codec uses JSON, so those
   two sequences become lists. `AdmissionMetadata.read` returns that decoded
   object. It cannot equal the retained in-memory admission under ordinary
   Python dict equality, even though all seven identity integers agree.
4. The first conjunct of original diagnostic lines 275–279 is
   `self.metadata.read(self.admission_ref) == self.admission`. It evaluates
   false. Python `and` therefore does not evaluate the remaining witness,
   source-map digest, descriptor-plan digest, lease, or process-owner checks.
   This failure is not evidence that any of those later checks failed.

The actual saved admission was decoded with the original codec. A source-
derived fixture restored only those two list fields to tuples; re-encoding it
reproduced the actual saved wire byte for byte. Executing the original three
adapter constructors and prerequisite methods with mock ports reproduced the
same failure. Digest and process callbacks were set to raise if called and
were not called. This fixture is not a recovered process heap; the tuple types
are established by the frozen executed source, corroborated by the saved wire.

No SDK child starts along this path. Original diagnostic constructor lines
266–271 sets child/owner/argv/terminal to None, state to `new`, child_started,
receipt_saved and receipt_attempted to false. Observed constructor lines 55–70
adds native_ports/observer/observed=None, gate=None and prerequisites_checked
false. Source adapter constructor lines 21–24 only retains the literal/ref.
Prerequisites fail before outer v7 line 164 calls `ports.start()`; the actual
SDK `Popen` is inside source adapter `start`, line 42. No observer is constructed.

Cleanup uses the original implementation:

- Observed adapter v4 lines 180–185 delegates to the original `drain_owned`
  because native_ports is None.
- Diagnostic v1 lines 355–363 sees child is None and skips stdin close,
  drain_group, child.wait, and _record_drained. It returns exactly
  `{physically_drained: true, no_services_started: true,
  exit_receipt_saved: false, child_started: false}`. State remains `new`,
  terminal remains None, and no child exit receipt is manufactured.
- Outer v7 lines 180–195 cancels its alarm, obtains this proof, then invokes
  original support `release_after_physical_drain` at line 194. Support v3
  lines 72–76 requires a true physical-drain field then closes that fd.
  Executing the original finally AST with mock alarm/close ports confirmed
  one alarm cancellation and one close after the no-child proof.

The focused source/mock check is
`check_minicpm5_formal_r4_prerequisite_failure_v1.py`; actual check terminal
f2bea4, exit 0. It authenticates seven reviewed sources against the frozen r4
complete map and blocks ML native imports plus ctypes/_ctypes. It neither
imports the historical preparation graph nor observes current processes,
services or locks. Root separately performs actual post-exit scans, original
lock reacquisition/release and two physical drain observations.

This run supplies no SDK import completion, observer result, scientific row,
native ABI qualification, full-model qualification or new speedup. The older
generic completion-auditor draft remains untested and unused. No frozen runtime
source was changed. A future minimal fix should make the new admission fields
JSON-stable before the unchanged original equality check, with the actual tuple
fixture in its regression check; this review does not implement or launch it.
