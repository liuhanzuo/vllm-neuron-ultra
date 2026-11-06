# SPDX-License-Identifier: Apache-2.0
"""LFM3-only artifact admission, outside the generic recompilation fallback."""
import json
import logging
import os
import concurrent.futures
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from .identity import (ARTIFACT_MANIFEST, AdmissionError, build_identity,
                       canonical_configuration, digest_file, read_json, require)

logger = logging.getLogger(__name__)


def artifact_files(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        require(not path.is_symlink(), f"Compiled artifact symlink unsupported: {path}")
        if path.is_file() and path != Path(directory) / ARTIFACT_MANIFEST:
            result[str(path.relative_to(directory))] = digest_file(path)
    require("model.pt" in result and "neuron_config.json" in result,
            "LFM3 artifact lacks its compiled model or config")
    return result


def admit_saved_configuration(directory, identity):
    """Compare the complete saved body; only replace our embedded metadata.

    NxDI InferenceConfig.save uses the same to_json_string serialization as
    the pre-build identity. Per-stage bucket/batch changes occur on deep copies
    of that config. No other saved field is normalized or ignored here.
    """
    saved = read_json(Path(directory) / "neuron_config.json")
    require(saved.get("metadata") == {"lfm3_ultra": identity},
            "Compiled config does not bind the admitted LFM3 identity")
    expected = identity.get("configuration")
    require(isinstance(expected, dict) and "metadata" in expected
            and expected["metadata"] is None,
            "Admitted LFM3 configuration must have original null metadata")
    saved_body = dict(saved)
    saved_body["metadata"] = None
    # Type-tagged comparison deliberately preserves the SDK's one documented
    # positive-Infinity default, rejects other nonfinites, and has no sentinel
    # collision with user-supplied strings/arrays/objects.
    require(canonical_configuration(saved_body) == canonical_configuration(expected),
            "Saved LFM3 configuration body differs from the admitted configuration")


def admit_existing(directory, identity):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "LFM3 artifacts must be an explicit ordinary directory")
    manifest = read_json(directory / ARTIFACT_MANIFEST)
    require(set(manifest) == {"complete", "identity", "files"}
            and manifest["complete"] is True,
            "LFM3 artifact manifest is incomplete")
    require(manifest["identity"] == identity,
            "LFM3 source, variant, model, scope or SDK identity mismatch")
    require(manifest["files"] == artifact_files(directory),
            "LFM3 compiled artifact bytes changed")
    admit_saved_configuration(directory, identity)


@contextmanager
def serial_compiler():
    """Limit only the SDK builder's submission pool during this model build."""
    import neuronx_distributed.trace.model_builder as builder
    original = builder.concurrent
    class OneCompiler(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, max_workers=None, *args, **kwargs):
            super().__init__(1, *args, **kwargs)
    class FuturesProxy:
        ThreadPoolExecutor = OneCompiler
        def __getattr__(self, name):
            return getattr(concurrent.futures, name)
    builder.concurrent = SimpleNamespace(futures=FuturesProxy())
    try:
        yield
    finally:
        builder.concurrent = original


def load_or_compile(owner, model_path, model_class, config, kwargs):
    """Return a loaded model without entering generic path deletion/fallback.

    Missing artifacts require an additional explicit compile opt-in. Existing
    incomplete/mismatched directories are never deleted, repaired or rebuilt.
    """
    raw = os.environ.get("NEURON_COMPILED_ARTIFACTS")
    require(bool(raw), "LFM3 requires a separate NEURON_COMPILED_ARTIFACTS directory")
    directory = Path(raw).absolute()
    require(not directory.is_symlink(), "Artifact directory cannot be a symlink")
    configuration = json.loads(config.to_json_string())
    identity = build_identity(model_path, configuration)
    config.metadata = {"lfm3_ultra": identity}
    if directory.exists():
        admit_existing(directory, identity)
        # No ValueError catch here: load errors must not trigger recompilation.
        owner._load_compiled_model(str(directory), model_class, kwargs)
    else:
        require(os.environ.get("NEURON_LFM3_ALLOW_COMPILE", "0") == "1",
                "Artifact missing; use a fresh path and NEURON_LFM3_ALLOW_COMPILE=1 to build")
        require(directory.parent.is_dir(), "Create the artifact parent directory explicitly")
        # Exclusive mkdir is also the ownership boundary for concurrent starts.
        try:
            directory.mkdir(exist_ok=False)
        except OSError as exc:
            raise AdmissionError("Cannot exclusively create the new LFM3 artifact") from exc
        owner.model = model_class(model_path, config)
        with serial_compiler():
            owner.model.compile(str(directory))
        admit_saved_configuration(directory, identity)
        owner.model.load(str(directory))
        files = artifact_files(directory)
        manifest = {"complete": True, "identity": identity, "files": files}
        # Completion is published only after compile AND load. A failed build
        # is left without a manifest and cannot be mistaken for a reusable one.
        temporary = directory / (ARTIFACT_MANIFEST + ".tmp")
        with temporary.open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / ARTIFACT_MANIFEST)
    logger.info("LFM3 variant=%s; context=2048; source-bound artifact=%s",
                identity["variant"], directory)
    # Both paths already loaded the model. The generic caller must not compile
    # a second time based on its ordinary cache-miss boolean.
    return True, str(directory), config
