# Independent review: discovery v2 / adapter v4 successor wiring

No actionable source blocker found. This review covers source and small synthetic metadata only. It does not read the real Release/output/plan/catalog/keyring, perform cryptography or network calls, acquire locks, observe processes, admit a task or promote either failed prior run.

Complete-module AST comparison establishes the only changes from the preceding reviewed integration:

- Adapter v4 imports discovery v2, binds its exact SHA256, and authenticates the v2 source path with the same `stable_read` call. The existing compliance helper and all other code remain unchanged.
- Input binding v3 imports discovery v2 and retains every old pin, adding only discovery v2 and adapter v4. All six fixed pins match their actual source bytes.
- Worker v3, root v3, saved audit v3 and recorder v3 contain only the corresponding module/version wiring changes. Resource, descriptor, history, admission, evidence, failure, drain and external-exit logic are unchanged.

Actual imported identities confirm adapter/binding/saved auditor all use discovery v2; worker/root/recorder use binding v3; recorder uses saved audit v3; saved audit uses adapter v4; adapter still uses the reviewed compliance helper. The original v1 discovery and adapter v2 remain dependencies of the frozen signature parser and remain pinned. Adapter's twelve fixed URLs equal the preceding version. The complete future source map must additionally pin every actually imported successor, as enforced by the existing unchanged source/loaded-module checks.

An added complete synthetic replay combines all 18 reported DEP-11 `@2` icon-name shapes in each of the three Releases and one exact compliance-23 annotation in each of the three signature status streams. It traverses the real saved replay and discovery functions, all 19 events / 27 nested files, and checks all three signature parser calls. Each Release returns only the original four main/universe compressed/raw indexes, and downloads remain exactly the twelve original Release/signature/Packages.xz URLs. No icon URL is fetched. All saved input bytes remain unchanged; the replay fully reproduces the discovery output with no cryptographic subprocess, package-content qualification or closed-unresolved claim. Package Filename containing `@` still fails the original validator.

The test imports the corrected source-isolated v4 fixture, whose synthetic INPUT_PINS substitution applies to discovery v2 as intended; no original fixture JSON is read. Its audit hook rejects network/process/lock operations and non-module file reads. The earlier v3 test-fixture failure and actual failed runs remain historical evidence and are not reclassified.

`check_dory_release_successors_independent_20261111_v1.py`: 3 independent tests passed, actual tool call terminal chunk `def9fe`, exit 0, unittest 0.084 seconds. The methods cover complete AST differences, actual imports/source hashes/old pin retention, and the combined Release-@ plus compliance-23 full replay. No root-owned or frozen source file was edited. Concrete runtime planning and admission remain root responsibilities.
