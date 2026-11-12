Native v6 and serial v5 correct the r3 origin check for Torch's two virtual namespace singletons. The failed r3 remains a failed diagnostic: session 91611, terminal b64cb0, exit 1. Its required PJRT helper actually completed, but the parent failed in origins() on torch.classes before spawn.

The successor reuses the frozen qwen35_sampler_torch_namespace_sources_v1.py helper. It accepts only the exact torch.ops and torch.classes singleton objects when sys.modules, torch, and their defining modules agree; their original ModuleType subclasses, relative class labels, installed module origins, and original __init__/__getattr__ code must agree with the pinned factory source. It does not skip arbitrary Torch names. The original ordinary package/native origin checks still run for every other module.

The native source bundle contains a new torch_namespace_sources group for the exact torch/__init__.py, torch/_ops.py, and torch/_classes.py source refs. Its worker_sources includes the exact reused helper. The worker reads and verifies all four sources before use, then executes the authenticated inert helper bytes. Virtual entries retain the module-to-string result shape and use an explicit stable value such as authenticated-virtual-source:/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch/_classes.py. Parent and spawn still must produce identical complete origins maps.

The serial source delivery exports native_namespace_source_sha256 with all four paths. Merge that map without replacement into the complete current source map together with the static local closure and the native bundle's existing source groups. Serial v5 rejects any missing or changed namespace pin before admission. No current catalog, history overlay, concrete spec, review, running handoff, lock, service or native execution was created by this source delivery.

All required PJRT helper behavior and cleanup, NumPy optional-probe denial, native checks, tokenizer checks, sampler checks, fresh spawn, resources, complete History, physical scope, drain-before-release ordering and real parent/child result checks remain unchanged. Function/class AST comparisons verify these preservation claims. Native v5 and serial v4 frozen files remain unchanged.

Validation: 51 native-blocked source/mock tests passed under both the default interpreter and the exact original Python 3.10; 21 serial source/mock tests passed. The tests reproduce v5's relative-label failure, authenticate both unmodified namespace fixtures, reject 18 identity/factory mutations plus missing/changed factory/helper refs and bytes, and retain foreign-module rejection. These results do not claim a successful native preflight, completed spawn, model execution or performance result.

Frozen artifacts:

- Native bundle: qwen35-package-native-preflight-source-bundle-20261112-v6.json
- Serial delivery: qwen35-package-preflight-serial-source-delivery-20261112-v5.json
- Test evidence: qwen35-package-native-preflight-source-checks-20261112-v6.json
- Runtime/preparation source diff: qwen35-package-torch-namespace-source-diff-20261112-v6.patch
