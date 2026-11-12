"""Complete inherited text verification with one exact v4 envelope transport.

The immutable partition, alias transport, Python-byte fixtures and return
receipt retain the original SDK verifier semantics. No source module is patched.
"""
import hashlib

from .io import HOST_SOURCE_ENVELOPE, host_source_bytes


def text_bytes(sources, path, expected):
    if str(path) == HOST_SOURCE_ENVELOPE[0]:
        return host_source_bytes(path, expected)
    return sources.text(path, expected)


def verify_text_sources(partition, guard):
    """Source-only path: no historical raw bytes or native libraries opened."""
    from research import minicpm5_layer1_operator_sdk_sources_v1 as sources
    sources.require(callable(guard), 'Explicit observer/runtime guard'); guard(); sources.check_partition(partition)
    for path, sha in partition['categories']['text_sha256'].items():
        guard()
        actual = (sources.aliases.source_sha(path, partition['metadata_aliases'], sources.text)
            if path in partition['metadata_aliases'] else hashlib.sha256(text_bytes(sources, path, sha)).hexdigest())
        sources.require(actual == sha, 'Unchanged exact text source')
    for path, sha in partition['categories']['python_bytes_sha256'].items():
        guard(); sources.metadata.python_source_bytes(path, sha)
    guard()
    return dict(complete_text_verification=True, source_map_sha256=partition['source_map_sha256'],
        partition_sha256=partition['partition_sha256'], historical_payload_bytes_read=0, native_bytes_read=0)
